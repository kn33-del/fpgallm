"""T3 mlp_unit + vec_unit: RMSNorm / residual / SiLU*mul / argmax bit-exact vs golden, and the
full SwiGLU MLP block (GEMMs on the reference engine model) with mlp/vec port arbitration."""
import random

import numpy as np
import pyuvm

from llmtb import cfg, golden
from llmtb.bfm import CmdDriver, MemRdServer, MemWrServer, MmEngineModel, logical_addr, logical_beats, rv
from llmtb.mem import RegionMem, beat_to_bytes
from llmtb.uvm import BaseEnv, BaseTest, CmdItem


class T3Env(BaseEnv):
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
        self.mm = self.domain.add(MmEngineModel(d, "mm", self.mem, rng=mk("mm"),
                                                execute=lambda c: golden.mm_execute(self.mem, c)))

        def grab_argmax(t):
            t["result"] = rv(d.argmax_idx)
        self.add_agent("vec", CmdDriver(d, d.vec_cmd, d.vec_cmd_valid, d.vec_cmd_ready, d.vec_done,
                                        rng=mk("vec"), name="vec_cmd", on_done=grab_argmax))
        self.add_agent("mlp", CmdDriver(d, d.mlp_cmd, d.mlp_cmd_valid, d.mlp_cmd_ready, d.mlp_done,
                                        rng=mk("mlp"), name="mlp_cmd"))


def cmp(name, got, want, errs, limit=4):
    got, want = np.asarray(got, np.int64), np.asarray(want, np.int64)
    bad = np.nonzero(got != want)[0]
    for b in bad[:limit]:
        errs.append(f"{name}[{b}] got {got[b]} want {want[b]}")
    if len(bad) > limit:
        errs.append(f"{name}: {len(bad)}/{len(want)} mismatches")


class T3Base(BaseTest):
    env_cls = T3Env

    def build_phase(self):
        super().build_phase()
        self.slot = 0

    def npr(self):
        return np.random.default_rng(self.env.rng.randrange(1 << 30))

    def next_offs(self):
        self.slot += 1
        return self.slot * 0x10000

    def vec_item(self, op, n, **kw):
        env, g = self.env, self.npr()
        base = self.next_offs()
        act, scr, hbm = env.mem[cfg.RGN_ACT], env.mem[cfg.RGN_SCRATCH], env.mem[cfg.RGN_HBM]
        lab = {cfg.V_RMSNORM: "rmsnorm", cfg.V_RESADD: "resadd", cfg.V_SILU_MUL: "silu_mul",
               cfg.V_ARGMAX: "argmax"}[op] + f" len={n}"
        f = dict(op=op, len=n, src0=cfg.addr(cfg.RGN_ACT, base), src1=0, dst=cfg.addr(cfg.RGN_SCRATCH, base))
        if op == cfg.V_RMSNORM:
            x = g.integers(-kw.get("xmax", 3000), kw.get("xmax", 3000) + 1, n)
            gm = g.integers(-6000, 6001, n)
            act.write_np(base, x, np.int16)
            hbm.write_np(base, gm, np.int16)
            f["src1"] = cfg.addr(cfg.RGN_HBM, base)
            exp = golden.rmsnorm(x, gm)
            chk = lambda it: (lambda e: (cmp("rmsnorm", scr.read_np(base, n, np.int8), exp, e), e)[1])([])
        elif op == cfg.V_RESADD:
            x = g.integers(-32768, 32768, n)
            a = g.integers(-32768, 32768, n)
            act.write_np(base, x, np.int16)
            act.write_np(base + 0x8000, a, np.int16)
            f["src1"] = cfg.addr(cfg.RGN_ACT, base + 0x8000)
            exp = golden.resadd(x, a)
            chk = lambda it: (lambda e: (cmp("resadd", scr.read_np(base, n, np.int16), exp, e), e)[1])([])
        elif op == cfg.V_SILU_MUL:
            tiles = cfg.ceil_div(n, 32)
            gu = g.integers(-kw.get("xmax", 2500), kw.get("xmax", 2500) + 1, tiles * 64)
            act.write_np(base, gu, np.int16)
            f["src1"] = cfg.addr(cfg.RGN_ACT, base + cfg.T3_UP_OFFSET)
            exp = golden.silu_mul_interleaved(gu, n)
            chk = lambda it: (lambda e: (cmp("silu_mul", scr.read_np(base, n, np.int8), exp, e), e)[1])([])
        else:
            x = g.integers(-30000, 30000, n)
            if kw.get("ties"):
                x[g.integers(0, n, 5)] = 32767       # several equal maxima -> lowest index wins
            act.write_np(base, x, np.int16)
            exp = golden.argmax(x)
            chk = lambda it: [] if it.result == exp else [f"argmax got {it.result} want {exp}"]
        return CmdItem("vec", cfg.VEC_CMD.pack(**f), f, chk, lab)

    def mlp_item(self, layer=0):
        env, g = self.env, self.npr()
        hbm, act = env.mem[cfg.RGN_HBM], env.mem[cfg.RGN_ACT]
        D, F = cfg.D_MODEL, cfg.D_FF
        x = g.integers(-128, 128, D)
        Wg = g.integers(-128, 128, (D, F), dtype=np.int8)
        Wu = g.integers(-128, 128, (D, F), dtype=np.int8)
        Wd = g.integers(-128, 128, (F, D), dtype=np.int8)
        sg, su = g.integers(300, 1200, F), g.integers(300, 1200, F)
        sd = g.integers(800, 2000, D)
        Wgu = np.empty((D, 2 * F), np.int8)
        sgu = np.empty(2 * F, np.int64)
        for t in range(F // 32):
            Wgu[:, 64 * t:64 * t + 32], Wgu[:, 64 * t + 32:64 * t + 64] = Wg[:, 32 * t:32 * t + 32], Wu[:, 32 * t:32 * t + 32]
            sgu[64 * t:64 * t + 32], sgu[64 * t + 32:64 * t + 64] = sg[32 * t:32 * t + 32], su[32 * t:32 * t + 32]
        w_off, s_off, x_off, o_off = 0x4000_0000, 0x7000_0000, 0x2000, 0x6000
        hbm.write(w_off, golden.pack_b_tiles(Wgu))
        hbm.write(w_off + cfg.T3_WD_OFFSET * cfg.N_HBM, golden.pack_b_tiles(Wd))
        hbm.write(s_off, golden.pack_scales(sgu))
        hbm.write(s_off + cfg.T3_SD_OFFSET, golden.pack_scales(sd))
        act.write_np(x_off, x, np.int8)
        gu = golden.gemm(x[None], Wgu, sgu, cfg.OUT_VEC)[0]
        u8 = golden.silu_mul_interleaved(gu, F)
        out = golden.gemm(u8[None], Wd, sd, cfg.OUT_VEC)[0]
        f = dict(layer=layer, x_base=cfg.addr(cfg.RGN_ACT, x_off), w_base=cfg.addr(cfg.RGN_HBM, w_off // cfg.N_HBM),
                 s_base=cfg.addr(cfg.RGN_HBM, s_off), out_base=cfg.addr(cfg.RGN_ACT, o_off))
        mm0 = len(env.mm.cmds)

        def chk(it):
            e = []
            cmp("silu_mul(u8)", env.mem[cfg.RGN_SCRATCH].read_np(cfg.T3_GU_BYTES, F, np.int8), u8, e)
            cmp("mlp_out", act.read_np(o_off, D, np.int16), out, e)
            cmds = env.mm.cmds[mm0:]
            if len(cmds) != 2:
                e.append(f"{len(cmds)} mm cmds, want 2")
            else:
                if (cmds[0]["k"], cmds[0]["n"]) != (D, 2 * F) or (cmds[1]["k"], cmds[1]["n"]) != (F, D):
                    e.append(f"GEMM dims {[(c['k'], c['n']) for c in cmds]}")
                if cmds[0]["tag"] == cmds[1]["tag"]:
                    e.append("gate/up and down GEMMs share a tag")
            return e
        return CmdItem("mlp", cfg.MLP_CMD.pack(**f), f, chk, f"mlp layer={layer}")


@pyuvm.test()
class T3RmsNorm(T3Base):
    async def body(self):
        await self.play("vec", [self.vec_item(cfg.V_RMSNORM, cfg.D_MODEL),
                                self.vec_item(cfg.V_RMSNORM, 100),
                                self.vec_item(cfg.V_RMSNORM, 1),
                                self.vec_item(cfg.V_RMSNORM, 333, xmax=32767)])


@pyuvm.test()
class T3ResAdd(T3Base):
    """Saturating residual add incl. overflow both ways; lengths not a multiple of 16."""
    async def body(self):
        await self.play("vec", [self.vec_item(cfg.V_RESADD, cfg.D_MODEL), self.vec_item(cfg.V_RESADD, 37),
                                self.vec_item(cfg.V_RESADD, 1)])


@pyuvm.test()
class T3SiluMul(T3Base):
    async def body(self):
        await self.play("vec", [self.vec_item(cfg.V_SILU_MUL, 100), self.vec_item(cfg.V_SILU_MUL, 32),
                                self.vec_item(cfg.V_SILU_MUL, 7, xmax=32767), self.vec_item(cfg.V_SILU_MUL, 512)])


@pyuvm.test()
class T3Argmax(T3Base):
    async def body(self):
        await self.play("vec", [self.vec_item(cfg.V_ARGMAX, 1000, ties=True), self.vec_item(cfg.V_ARGMAX, 17),
                                self.vec_item(cfg.V_ARGMAX, 1), self.vec_item(cfg.V_ARGMAX, cfg.VOCAB)])


@pyuvm.test()
class T3MlpBlock(T3Base):
    """Full SwiGLU block at model dims: gate/up GEMM -> SiLU*mul -> down GEMM."""
    async def body(self):
        it = self.mlp_item(layer=2)
        await self.play("mlp", [it])
        self.profile.add("mlp_block", cycles=it.done_cycle - it.acc_cycle)


@pyuvm.test()
class T3MlpWhileVec(T3Base):
    """Arbitration: llm_ctrl vec commands issued while an MLP block owns the vec unit."""
    async def body(self):
        import cocotb
        mlp = self.mlp_item(layer=1)
        vecs = [self.vec_item(cfg.V_RESADD, 200), self.vec_item(cfg.V_ARGMAX, 300)]
        t = cocotb.start_soon(self.play("mlp", [mlp]))
        await self.env.domain.cycles(30)
        await self.play("vec", vecs)
        await t
