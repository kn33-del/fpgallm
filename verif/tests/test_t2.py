"""T2 attn_unit: full decode attention for one layer against a bit-exact golden.

The matrix engine is the Python reference model (golden.mm_execute) so this bench isolates
T2: GQA command order/addresses, RoPE, K/V append payloads, softmax, head concat, Wo.
The K/V write server emulates T4's kv_cache_mgr per the llm_pkg KV contract (kv_tag_t
requests, kv_pc_off placement). Wide addresses are per-PC bytes; the unit memory stores a
tile contiguously at logical byte = per-PC byte * N_HBM.
"""
import math
import random

import numpy as np
import pyuvm

from llmtb import cfg, golden
from llmtb.bfm import MemRdServer, MemWrServer, MmEngineModel, CmdDriver, logical_addr, logical_beats
from llmtb.mem import RegionMem, beat_to_bytes
from llmtb.uvm import BaseEnv, BaseTest, CmdItem

H, KVH, HD = cfg.N_HEADS, cfg.N_KV_HEADS, cfg.HEAD_DIM
GROUP = H // KVH


class T2Env(BaseEnv):
    def make_bfms(self):
        d = self.dut
        self.mem = RegionMem()
        self.rng = random.Random(self.seed)
        mk = lambda n: random.Random(hash((self.seed, n)))
        self.rd = self.domain.add(MemRdServer(d, "rd", cfg.BEAT_W, lambda r: logical_beats(self.mem, r, cfg.BEAT_W),
                                              rng=mk("rd"), name="rd"))

        def wbeat(req, idx, data):
            rgn, off = logical_addr(req, idx, cfg.BEAT_W)
            self.mem[rgn].write(off, beat_to_bytes(data, 32))
        self.wr = self.domain.add(MemWrServer(d, "wr", cfg.BEAT_W, wbeat, rng=mk("wr"), name="wr"))
        self.kv = self.domain.add(MemWrServer(d, "kv", cfg.BEAT_W, lambda *a: None, rng=mk("kv"), name="kv_wr"))
        self.kv.on_complete.append(self.kv_scatter)
        self.mm = self.domain.add(MmEngineModel(d, "mm", self.mem, rng=mk("mm"),
                                                execute=lambda c: golden.mm_execute(self.mem, c)))
        self.add_agent("attn", CmdDriver(d, d.cmd, d.cmd_valid, d.cmd_ready, d.done, rng=mk("cmd"), name="attn_cmd"))

    def kv_scatter(self, req, beats):
        """T4 kv_cache_mgr emulation: base.off[28:0] = kv_tag_t, payload = one K or V vector."""
        rgn, off = cfg.addr_split(req["base"])
        if rgn != cfg.RGN_HBM or len(beats) != cfg.HEAD_BEATS:
            return                                  # contract test reports it
        t = cfg.KV_TAG.unpack(off & cfg.PC_MASK)
        vec = np.frombuffer(b"".join(beat_to_bytes(b, 32) for b in beats), np.int8)
        base = kv_logical(t["layer"], t["kvh"], t["is_v"])
        pos, hbm = t["pos"], self.mem[cfg.RGN_HBM]
        for d_ in range(HD):
            if t["is_v"]:   # V[pos][d]: tile (k=pos/32, n=d/32), row pos%32, col d%32
                tile, r, c = (d_ // 32) * (cfg.MAX_SEQ // 32) + pos // 32, pos % 32, d_ % 32
            else:           # K^T[d][pos]: tile (k=d/32, n=pos/32), row d%32, col pos%32
                tile, r, c = (pos // 32) * (HD // 32) + d_ // 32, d_ % 32, pos % 32
            hbm.write(base + tile * 1024 + r * 32 + c, vec[d_:d_ + 1].tobytes())


def kv_logical(layer, kvh, is_v):
    return cfg.kv_pc_off(layer, kvh, is_v) * cfg.N_HBM


def rope_table(pos, theta=500000.0):
    P = HD // 2
    inv = [theta ** (-2 * i / HD) for i in range(P)]
    c = [max(-32768, min(32767, round(math.cos(pos * f) * 32768))) for f in inv]
    s = [max(-32768, min(32767, round(math.sin(pos * f) * 32768))) for f in inv]
    return np.array(c, np.int64), np.array(s, np.int64)


class T2Attn(BaseTest):
    env_cls = T2Env
    pos_list = [37]
    layer = 3

    def setup_layer(self, layer, pos):
        env = self.env
        npr = np.random.default_rng(env.rng.randrange(1 << 30))
        hbm = env.mem[cfg.RGN_HBM]
        Q = cfg.QKV_ELEMENTS
        w_off, s_off = 0x1000_0000, 0x2000_0000
        x_off, out_off = 0x1000, 0x8000
        x = npr.integers(-128, 128, cfg.D_MODEL)
        Wqkv = npr.integers(-128, 128, (cfg.D_MODEL, Q))
        Wo = npr.integers(-128, 128, (cfg.D_MODEL, cfg.D_MODEL))
        s_qkv = npr.integers(1000, 4000, Q)
        s_wo = npr.integers(1500, 3500, cfg.D_MODEL)
        env.mem[cfg.RGN_ACT].write_np(x_off, x, np.int8)
        hbm.write(w_off, golden.pack_b_tiles(Wqkv))
        hbm.write(w_off + cfg.WO_OFFSET * cfg.N_HBM, golden.pack_b_tiles(Wo))
        hbm.write(s_off, golden.pack_scales(s_qkv))
        hbm.write(s_off + cfg.WO_SCALE_OFFSET, golden.pack_scales(s_wo))
        # KV cache (logical layout), random history; KV scales
        cache_k = npr.integers(-128, 128, (KVH, HD, cfg.MAX_SEQ))   # K^T
        cache_v = npr.integers(-128, 128, (KVH, cfg.MAX_SEQ, HD))
        ks = npr.integers(500, 3000, (KVH, cfg.MAX_SEQ))
        vs = npr.integers(300, 800, (KVH, HD))
        for kvh in range(KVH):
            pair = layer * KVH + kvh
            hbm.write(kv_logical(layer, kvh, 0), golden.pack_b_tiles(cache_k[kvh]))
            hbm.write(kv_logical(layer, kvh, 1), golden.pack_b_tiles(cache_v[kvh]))
            sb = cfg.T2_KV_SCALE_BASE + pair * cfg.KV_SCALE_BYTES
            hbm.write(sb, golden.pack_scales(ks[kvh]))
            hbm.write(sb + cfg.K_SCALE_BYTES, golden.pack_scales(vs[kvh]))
        cos, sin = rope_table(pos)
        hbm.write_np(cfg.T2_ROPE_BASE + pos * cfg.COEFF_BYTES, np.concatenate([cos, sin]), np.int16)
        # ---------------- golden ----------------
        qkv = golden.gemm(x[None], Wqkv, s_qkv, cfg.OUT_VEC)[0]
        qkv8 = np.zeros(Q, np.int64)
        for h in range(H + 2 * KVH):
            qkv8[h * HD:(h + 1) * HD] = golden.rope_head(qkv[h * HD:(h + 1) * HD], cos, sin, pos, h < H + KVH)
        kn = qkv8[H * HD:(H + KVH) * HD].reshape(KVH, HD)
        vn = qkv8[(H + KVH) * HD:].reshape(KVH, HD)
        ck, cv = cache_k.copy(), cache_v.copy()
        ck[:, :, pos] = kn
        cv[:, pos, :] = vn
        heads = np.zeros(cfg.D_MODEL, np.int64)
        probs_last = None
        for h in range(H):
            g = h // GROUP
            sc = golden.gemm(qkv8[h * HD:(h + 1) * HD][None], ck[g][:, :pos + 1], ks[g][:pos + 1], cfg.OUT_VEC)[0]
            p = golden.softmax_i8(sc)
            heads[h * HD:(h + 1) * HD] = golden.gemm(p[None], cv[g][:pos + 1, :], vs[g], cfg.OUT_I8)[0]
            probs_last = p
        out = golden.gemm(heads[None], Wo, s_wo, cfg.OUT_VEC)[0]
        f = dict(layer=layer, pos=pos, x_base=cfg.addr(cfg.RGN_ACT, x_off), w_base=cfg.addr(cfg.RGN_HBM, w_off // cfg.N_HBM),
                 s_base=cfg.addr(cfg.RGN_HBM, s_off), out_base=cfg.addr(cfg.RGN_ACT, out_off))
        exp = dict(qkv8=qkv8, kn=kn, vn=vn, heads=heads, out=out, probs_last=probs_last)
        return f, exp

    def make_check(self, f, exp, mm_start, kv_start):
        env, pos, layer = self.env, f["pos"], f["layer"]

        def cmp(name, got, want, errs, limit=4):
            bad = np.nonzero(np.asarray(got) != np.asarray(want))[0]
            for b in bad[:limit]:
                errs.append(f"{name}[{b}] got {got[b]} want {want[b]}")
            if len(bad) > limit:
                errs.append(f"{name}: {len(bad)} mismatches total")

        def check(it):
            errs = []
            scr, act = env.mem[cfg.RGN_SCRATCH], env.mem[cfg.RGN_ACT]
            cmp("qkv_i8(rope)", scr.read_np(cfg.T2_QKV_I8, cfg.QKV_ELEMENTS, np.int8).astype(np.int64), exp["qkv8"], errs)
            cmp("heads", scr.read_np(cfg.T2_HEADS, cfg.D_MODEL, np.int8).astype(np.int64), exp["heads"], errs)
            p = scr.read_np(cfg.T2_PROBS, cfg.ceil_div(pos + 1, 32) * 32, np.int8).astype(np.int64)
            cmp("probs(last head)", p[:pos + 1], exp["probs_last"], errs)
            if np.any(p[pos + 1:] != 0):
                errs.append("softmax padding bytes nonzero")
            _, ooff = cfg.addr_split(f["out_base"])
            cmp("attn_out", act.read_np(ooff, cfg.D_MODEL, np.int16).astype(np.int64), exp["out"], errs)
            # ---- matrix command stream: QKV, (score, value) x H, Wo
            cmds = env.mm.cmds[mm_start:]
            if len(cmds) != 2 + 2 * H:
                errs.append(f"{len(cmds)} mm cmds, want {2 + 2 * H}")
            else:
                if cmds[0]["k"] != cfg.D_MODEL or cmds[0]["n"] != cfg.QKV_ELEMENTS:
                    errs.append(f"QKV cmd dims {cmds[0]['k']}x{cmds[0]['n']}")
                for h in range(H):
                    sc, vc = cmds[1 + 2 * h], cmds[2 + 2 * h]
                    kb = cfg.addr(cfg.RGN_HBM, cfg.kv_pc_off(layer, h // GROUP, 0))
                    if sc["b_base"] != kb or sc["n"] != pos + 1 or sc["k"] != HD:
                        errs.append(f"head {h} score cmd b_base/n/k wrong (GQA mapping)")
                    if vc["k"] != pos + 1 or vc["n"] != HD or vc["b_row_stride"] != cfg.V_N_TILE_STRIDE:
                        errs.append(f"head {h} value cmd k/n/stride wrong")
            # ---- K/V append: one request per vector, contents = rotated K / plain V
            kvr = env.kv.reqs[kv_start:]
            if len(kvr) != 2 * KVH:
                errs.append(f"{len(kvr)} kv requests, want {2 * KVH}")
            for kvh in range(KVH):
                for is_v, want in ((0, exp["kn"][kvh]), (1, exp["vn"][kvh])):
                    base = kv_logical(layer, kvh, is_v)
                    if is_v:
                        got = [env.mem[cfg.RGN_HBM].read_np(base + ((d // 32) * (cfg.MAX_SEQ // 32) + pos // 32) * 1024
                                                            + (pos % 32) * 32 + d % 32, 1, np.int8)[0] for d in range(HD)]
                    else:
                        got = [env.mem[cfg.RGN_HBM].read_np(base + ((pos // 32) * (HD // 32) + d // 32) * 1024
                                                            + (d % 32) * 32 + pos % 32, 1, np.int8)[0] for d in range(HD)]
                    cmp(f"kv_append kvh={kvh} {'V' if is_v else 'K'}", np.array(got, np.int64), want, errs)
            return errs
        return check

    async def body(self):
        for pos in self.pos_list:
            f, exp = self.setup_layer(self.layer, pos)
            it = CmdItem("attn", cfg.ATTN_CMD.pack(**f), f, None, f"attn layer={self.layer} pos={pos}")
            it.check = self.make_check(f, exp, len(self.env.mm.cmds), len(self.env.kv.reqs))
            await self.play("attn", [it])
            self.profile.add(f"attn:pos{pos}", cycles=it.done_cycle - it.acc_cycle,
                             mm_cmds=len(self.env.mm.cmds), kv_reqs=len(self.env.kv.reqs))


@pyuvm.test()
class T2AttnPos0(T2Attn):
    """Position 0: RoPE identity, single-element softmax."""
    pos_list = [0]


@pyuvm.test()
class T2AttnPartialTile(T2Attn):
    """pos=37: ragged score N (38) and value K (38) edges; GQA over all heads."""
    pos_list = [37]


@pyuvm.test()
class T2AttnLastSlot(T2Attn):
    """Largest configured prefix: pos = MAX_SEQ-1."""
    pos_list = [cfg.MAX_SEQ - 1]
    layer = cfg.N_LAYERS - 1


# ------------------------------------------------------------------------------------------
# Contract checks against llm_pkg (cross-team). These describe what T4 implements.
# ------------------------------------------------------------------------------------------
@pyuvm.test()
class T2ContractKvWrite(T2Attn):
    """llm_pkg KV CONTRACT: each kv_wr request = ONE vector (K or V), base.off[28:0] = kv_tag_t,
    row_beats = HEAD_DIM/32, rgn = HBM. (T4 kv_cache_mgr decodes exactly this.)"""
    pos_list = [5]

    def extra_checks(self):
        errs = []
        reqs = self.env.kv.reqs
        if len(reqs) != 2 * KVH:
            errs.append(f"T2 issued {len(reqs)} kv_wr requests per token, contract needs {2 * KVH} "
                        f"(one K + one V per kv head)")
        for r in reqs[:4]:
            rgn, off = cfg.addr_split(r["base"])
            tag = cfg.KV_TAG.unpack(off & ((1 << 29) - 1))
            if r["row_beats"] != HD // 32 or tag["pos"] != 5 or tag["layer"] != self.layer:
                errs.append(f"kv req off={off:#x} row_beats={r['row_beats']} decodes as kv_tag "
                            f"{tag} (want layer={self.layer} pos=5, row_beats={HD // 32})")
        return errs


@pyuvm.test()
class T2ContractKvRead(T2Attn):
    """llm_pkg KV CONTRACT: score/value GEMM b_base.off[28:0] = kv_pc_off(layer,kvh,is_v),
    V row_stride = (MAX_SEQ/32)*32 per-PC bytes (T4 weight_streamer addressing)."""
    pos_list = [3]

    def extra_checks(self):
        errs = []
        cmds = self.env.mm.cmds
        sc, vc = cmds[1], cmds[2]
        _, soff = cfg.addr_split(sc["b_base"])
        _, voff = cfg.addr_split(vc["b_base"])
        want_k, want_v = cfg.kv_pc_off(self.layer, 0, 0), cfg.kv_pc_off(self.layer, 0, 1)
        if soff & ((1 << 29) - 1) != want_k:
            errs.append(f"score b_base.off={soff:#x}; streamer uses off[28:0]={soff & ((1 << 29) - 1):#x}, "
                        f"contract kv_pc_off={want_k:#x}")
        if voff & ((1 << 29) - 1) != want_v:
            errs.append(f"value b_base.off={voff:#x}; contract kv_pc_off={want_v:#x}")
        if vc["b_row_stride"] != (cfg.MAX_SEQ // 32) * 32:
            errs.append(f"value b_row_stride={vc['b_row_stride']}, contract per-PC stride {(cfg.MAX_SEQ // 32) * 32}")
        return errs
