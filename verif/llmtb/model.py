"""Host side of the full pipeline: random W8A8 model, HBM packer (llm_pkg contracts), program
builder, and a bit-exact golden decode step composed from llmtb.golden.

HBM placement (flat 34-bit address = {PC[4:0], off[28:0]}):
  wide (B operands, per-PC offsets, tile t row r -> PC r @ off + t*32), allocated in order from
      0x0010_0000: per layer attn ([Wq|Wk|Wv] then Wo at +WO_OFFSET), mlp ([Wg|Wu] then Wd at
      +WD_OFFSET); then the LM head. Must stay below 0x0400_0000 (T2 KV scales in PC4).
      KV cache @ kv_pc_off(layer, kvh, is_v) = 0x1000_0000+ (written by T4 kv_cache_mgr)
  narrow (full addresses):
      PC 2 from 0x0800_0000: scales, gammas, program
      PC 3 at 0x1100_0000: embedding table (past the KV cache; 131 MB for TinyLlama)
      T2 fixed params: KV scales @ 0x0_8400_0000, RoPE table @ 0x0_8500_0000
"""
import math

import numpy as np

from . import cfg, golden

ACT_X, ACT_H, ACT_A = 0x0000, 0x1000, 0x2000           # ACT: residual x (i16), normed h (i8), a/m (i16)
SCR_LOGITS = 0x10000                                   # SCRATCH: logits (i16), above T2/T3 scratch
T2_KV_SCALE_BASE, T2_ROPE_BASE = 0x0_8400_0000, 0x0_8500_0000
H, KVH, HD, D, F = cfg.N_HEADS, cfg.N_KV_HEADS, cfg.HEAD_DIM, cfg.D_MODEL, cfg.D_FF
Q = cfg.QKV_ELEMENTS


def put_wide(mem, pc_off, W):
    """Contract tile layout for B = W[K x N]: n_tile outer, k_tile inner; tile row r -> PC r."""
    K, N = W.shape
    kt, nt = cfg.ceil_div(K, 32), cfg.ceil_div(N, 32)
    Wp = np.zeros((kt * 32, nt * 32), np.int8)
    Wp[:K, :N] = W
    # [k_tile i, row r, n_tile j, col c] -> PC r holds, in tile order (j outer, i inner), row r
    per_pc = Wp.reshape(kt, 32, nt, 32).transpose(1, 2, 0, 3)
    for r in range(32):
        mem.write((r << 29) | pc_off, np.ascontiguousarray(per_pc[r]).tobytes())
    return kt * nt * 32                                # per-PC bytes used


class Alloc:
    def __init__(self, base):
        self.p = base

    def take(self, nbytes, align=4096):
        a = self.p
        self.p += (nbytes + align - 1) // align * align
        return a


def rope_table(pos, theta=500000.0):
    """RoPE row: HEAD_DIM/2 cos then sin, int16 Q15 (T2 format)."""
    inv = [theta ** (-2 * i / HD) for i in range(HD // 2)]
    c = [max(-32768, min(32767, round(math.cos(pos * f) * 32768))) for f in inv]
    s = [max(-32768, min(32767, round(math.sin(pos * f) * 32768))) for f in inv]
    return np.array(c, np.int64), np.array(s, np.int64)


class TinyModel:
    qm, qs, theta = 1, 0, 500000.0       # T3 QUANT_MULT/SHIFT (must match RTL), RoPE theta
    mm_shift = 16                        # T1 REQUANT_SHIFT (must match RTL)

    def __init__(self, seed=1):
        g = np.random.default_rng(seed)
        r8 = lambda *shape: g.integers(-128, 128, shape, dtype=np.int8)
        ri = lambda lo, hi, n: g.integers(lo, hi, n)
        L = cfg.N_LAYERS
        self.emb = g.integers(-2000, 2001, (cfg.VOCAB, D))
        self.g_attn = [ri(1500, 3000, D) * g.choice([-1, 1], D) for _ in range(L)]
        self.g_mlp = [ri(1500, 3000, D) * g.choice([-1, 1], D) for _ in range(L)]
        self.g_final = ri(1500, 3000, D)
        self.Wqkv = [r8(D, Q) for _ in range(L)]
        self.s_qkv = [ri(25000, 45000, Q) for _ in range(L)]
        self.Wo = [r8(D, D) for _ in range(L)]
        self.s_wo = [ri(3000, 6000, D) for _ in range(L)]
        self.ks = [[ri(5000, 9000, cfg.MAX_SEQ) for _ in range(KVH)] for _ in range(L)]
        self.vs = [[ri(1500, 2500, HD) for _ in range(KVH)] for _ in range(L)]
        self.Wg = [r8(D, F) for _ in range(L)]
        self.Wu = [r8(D, F) for _ in range(L)]
        self.sg = [ri(3000, 5000, F) for _ in range(L)]
        self.su = [ri(3000, 5000, F) for _ in range(L)]
        self.Wd = [r8(F, D) for _ in range(L)]
        self.sd = [ri(7000, 11000, D) for _ in range(L)]
        self.Wlm = r8(D, cfg.VOCAB)
        self.s_lm = ri(2000, 4000, cfg.VOCAB)
        # golden KV cache state
        self.Kc = np.zeros((L, KVH, HD, cfg.MAX_SEQ), np.int64)
        self.Vc = np.zeros((L, KVH, cfg.MAX_SEQ, HD), np.int64)

    # ---------------------------------------------------------------- interleaved gate/up
    def gu(self, l):
        Wgu = np.empty((D, 2 * F), np.int8)
        sgu = np.empty(2 * F, np.int64)
        for t in range(F // 32):
            a, b = slice(64 * t, 64 * t + 32), slice(64 * t + 32, 64 * t + 64)
            c = slice(32 * t, 32 * t + 32)
            Wgu[:, a], Wgu[:, b], sgu[a], sgu[b] = self.Wg[l][:, c], self.Wu[l][:, c], self.sg[l][c], self.su[l][c]
        return Wgu, sgu

    # ---------------------------------------------------------------- packer
    def pack(self, hbm, positions):
        L = cfg.N_LAYERS
        al = Alloc((2 << 29) | 0x0800_0000)
        wide = Alloc(0x0010_0000)
        self.addr = A = {}

        def wsize(K, N):
            return cfg.ceil_div(K, 32) * cfg.ceil_div(N, 32) * 32

        def narrow(name, data):
            a = al.take(len(data))
            hbm.write(a, data)
            A[name] = a
        for l in range(L):
            A[f"w_attn{l}"] = wide.take(cfg.WO_OFFSET + wsize(D, D))
            put_wide(hbm, A[f"w_attn{l}"], self.Wqkv[l])
            put_wide(hbm, A[f"w_attn{l}"] + cfg.WO_OFFSET, self.Wo[l])
            narrow(f"s_attn{l}", golden.pack_scales(self.s_qkv[l]) + golden.pack_scales(self.s_wo[l]))
            assert len(golden.pack_scales(self.s_qkv[l])) == cfg.WO_SCALE_OFFSET
            A[f"w_mlp{l}"] = wide.take(cfg.T3_WD_OFFSET + wsize(F, D))
            Wgu, sgu = self.gu(l)
            put_wide(hbm, A[f"w_mlp{l}"], Wgu)
            put_wide(hbm, A[f"w_mlp{l}"] + cfg.T3_WD_OFFSET, self.Wd[l])
            sb = golden.pack_scales(sgu)
            assert len(sb) == cfg.T3_SD_OFFSET
            narrow(f"s_mlp{l}", sb + golden.pack_scales(self.sd[l]))
            narrow(f"g_attn{l}", self.g_attn[l].astype("<i2").tobytes())
            narrow(f"g_mlp{l}", self.g_mlp[l].astype("<i2").tobytes())
            for kvh in range(KVH):
                pair = l * KVH + kvh
                sb = T2_KV_SCALE_BASE + pair * cfg.KV_SCALE_BYTES
                hbm.write(sb, golden.pack_scales(self.ks[l][kvh]))
                hbm.write(sb + cfg.K_SCALE_BYTES, golden.pack_scales(self.vs[l][kvh]))
        narrow("g_final", self.g_final.astype("<i2").tobytes())
        A["emb"] = (3 << 29) | 0x1100_0000
        hbm.write(A["emb"], self.emb.astype("<i2").tobytes())
        A["w_lm"] = wide.take(wsize(D, cfg.VOCAB))
        put_wide(hbm, A["w_lm"], self.Wlm)
        assert wide.p <= 0x0400_0000, f"wide weights overflow into T2 KV-scale area ({wide.p:#x})"
        narrow("s_lm", golden.pack_scales(self.s_lm))
        for p in positions:
            c, s = rope_table(p, self.theta)
            hbm.write(T2_ROPE_BASE + p * cfg.COEFF_BYTES, np.concatenate([c, s]).astype("<i2").tobytes())
        prog = self.program()
        A["prog"] = al.take(len(prog) * 32)
        for i, ins in enumerate(prog):
            hbm.write(A["prog"] + 32 * i, cfg.INSTR.pack(**ins).to_bytes(32, "little"))
        self.n_instr = len(prog)
        return A

    def program(self):
        A, act, hbm, scr = self.addr, cfg.RGN_ACT, cfg.RGN_HBM, cfg.RGN_SCRATCH
        a = cfg.addr
        P = [dict(op=cfg.OP_EMBED, src0=a(hbm, A["emb"]), dst=a(act, ACT_X))]
        for l in range(cfg.N_LAYERS):
            P += [dict(op=cfg.OP_RMSNORM, src0=a(act, ACT_X), src1=a(hbm, A[f"g_attn{l}"]), dst=a(act, ACT_H)),
                  dict(op=cfg.OP_ATTN_BLOCK, layer=l, src0=a(act, ACT_H), src1=a(hbm, A[f"w_attn{l}"]),
                       src2=a(hbm, A[f"s_attn{l}"]), dst=a(act, ACT_A)),
                  dict(op=cfg.OP_RESADD, src0=a(act, ACT_X), src1=a(act, ACT_A), dst=a(act, ACT_X)),
                  dict(op=cfg.OP_RMSNORM, src0=a(act, ACT_X), src1=a(hbm, A[f"g_mlp{l}"]), dst=a(act, ACT_H)),
                  dict(op=cfg.OP_MLP_BLOCK, layer=l, src0=a(act, ACT_H), src1=a(hbm, A[f"w_mlp{l}"]),
                       src2=a(hbm, A[f"s_mlp{l}"]), dst=a(act, ACT_A)),
                  dict(op=cfg.OP_RESADD, src0=a(act, ACT_X), src1=a(act, ACT_A), dst=a(act, ACT_X))]
        P += [dict(op=cfg.OP_RMSNORM, src0=a(act, ACT_X), src1=a(hbm, A["g_final"]), dst=a(act, ACT_H)),
              dict(op=cfg.OP_GEMM, m=1, k=D, n=cfg.VOCAB, src0=a(act, ACT_H), src1=a(hbm, A["w_lm"]),
                   src2=a(hbm, A["s_lm"]), dst=a(scr, SCR_LOGITS)),
              dict(op=cfg.OP_ARGMAX, n=cfg.VOCAB, src0=a(scr, SCR_LOGITS)),
              dict(op=cfg.OP_END)]
        return P

    # ---------------------------------------------------------------- golden decode step
    def step(self, token, pos):
        t = {}
        x = self.emb[token].astype(np.int64)
        c, s = rope_table(pos, self.theta)
        qm, qs = self.qm, self.qs
        for l in range(cfg.N_LAYERS):
            h = golden.rmsnorm(x, self.g_attn[l], qm, qs)
            qkv = golden.gemm(h[None], self.Wqkv[l], self.s_qkv[l], cfg.OUT_VEC, shift=self.mm_shift)[0]
            q8 = np.zeros(Q, np.int64)
            for hh in range(H + 2 * KVH):
                q8[hh * HD:(hh + 1) * HD] = golden.rope_head(qkv[hh * HD:(hh + 1) * HD], c, s, pos, hh < H + KVH)
            for kvh in range(KVH):
                self.Kc[l, kvh, :, pos] = q8[(H + kvh) * HD:(H + kvh + 1) * HD]
                self.Vc[l, kvh, pos, :] = q8[(H + KVH + kvh) * HD:(H + KVH + kvh + 1) * HD]
            heads = np.zeros(D, np.int64)
            for hh in range(H):
                g = hh // (H // KVH)
                sc = golden.gemm(q8[hh * HD:(hh + 1) * HD][None], self.Kc[l, g][:, :pos + 1],
                                 self.ks[l][g][:pos + 1], cfg.OUT_VEC, shift=self.mm_shift)[0]
                p = golden.softmax_i8(sc)
                heads[hh * HD:(hh + 1) * HD] = golden.gemm(p[None], self.Vc[l, g][:pos + 1], self.vs[l][g], cfg.OUT_I8, shift=self.mm_shift)[0]
            att = golden.gemm(heads[None], self.Wo[l], self.s_wo[l], cfg.OUT_VEC, shift=self.mm_shift)[0]
            x = golden.resadd(x, att)
            h = golden.rmsnorm(x, self.g_mlp[l], qm, qs)
            Wgu, sgu = self.gu(l)
            gu = golden.gemm(h[None], Wgu, sgu, cfg.OUT_VEC, shift=self.mm_shift)[0]
            u = golden.silu_mul_interleaved(gu, F, qm, qs)
            m = golden.gemm(u[None], self.Wd[l], self.sd[l], cfg.OUT_VEC, shift=self.mm_shift)[0]
            x = golden.resadd(x, m)
            t[f"x_l{l}"] = x
        h = golden.rmsnorm(x, self.g_final, qm, qs)
        logits = golden.gemm(h[None], self.Wlm, self.s_lm, cfg.OUT_VEC, shift=self.mm_shift)[0]
        t.update(x=x, h=h, logits=logits, token=golden.argmax(logits))
        return t
