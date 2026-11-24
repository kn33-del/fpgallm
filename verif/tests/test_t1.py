"""T1 mm_engine: GEMV/GEMM correctness (all out formats, ragged edges, strided B, prefill M>1),
back-to-back tagged commands, and array-utilization profiling."""
import random

import numpy as np
import pyuvm

from llmtb import cfg, golden
from llmtb.bfm import MemRdServer, MemWrServer, MmMaster, logical_addr, logical_beats, wide_beats
from llmtb.mem import RegionMem, beat_to_bytes
from llmtb.perf import Profile
from llmtb.uvm import BaseEnv, BaseTest, CmdItem


class T1Env(BaseEnv):
    bp = dict(p_req=0.7, p_valid=0.8, p_ready=0.8, lat=(1, 6))

    def make_bfms(self):
        d, bp = self.dut, self.bp
        self.mem = RegionMem()
        self.rng = random.Random(self.seed)
        mk = lambda n: random.Random(hash((self.seed, n)))
        for nm, w in (("a", cfg.BEAT_W), ("s", cfg.BEAT_W), ("b", cfg.WSTREAM_W)):
            fn = (lambda req: wide_beats(self.mem, req)) if nm == "b" else                  (lambda req, w=w: logical_beats(self.mem, req, w))
            srv = MemRdServer(d, nm, w, fn, rng=mk(nm),
                              p_req=bp["p_req"], p_valid=bp["p_valid"], lat=bp["lat"], name=f"{nm}_rd")
            setattr(self, f"{nm}_rd", self.domain.add(srv))

        def wbeat(req, idx, data):
            rgn, off = logical_addr(req, idx, cfg.BEAT_W)
            self.mem[rgn].write(off, beat_to_bytes(data, 32))
        self.c_wr = self.domain.add(MemWrServer(d, "c", cfg.BEAT_W, wbeat, rng=mk("c"), p_req=bp["p_req"],
                                                p_ready=bp["p_ready"], name="c_wr"))
        self.mm = MmMaster(d, "cmd", rng=mk("mm"), p_valid=bp["p_valid"])
        self.add_agent("mm", self.mm)


class T1FastEnv(T1Env):
    bp = dict(p_req=1.0, p_valid=1.0, p_ready=1.0, lat=(0, 0))


class T1Base(BaseTest):
    env_cls = T1Env
    cases = []

    def build_phase(self):
        super().build_phase()
        self.gemm_prof = []

    def make_case(self, i, m, k, n, fmt, b_stride_pad=0, tag=None):
        env, rng = self.env, self.env.rng
        npr = np.random.default_rng(rng.randrange(1 << 30))
        kt, nt = cfg.ceil_div(k, 32), cfg.ceil_div(n, 32)
        A = npr.integers(-128, 128, (m, k), dtype=np.int64)
        W = npr.integers(-128, 128, (k, n), dtype=np.int64)
        S = npr.integers(-(1 << 14), 1 << 14, n, dtype=np.int64) * (1 if fmt == cfg.OUT_VEC else 1)
        if fmt == cfg.OUT_I8:
            S = npr.integers(-64, 64, n, dtype=np.int64)
        a_off, b_off, s_off, c_off = i * 0x40000, i * 0x400000, 0x8000000 + i * 0x10000, i * 0x40000
        # A rows padded to whole K tiles; fill padding with garbage (RTL must mask it)
        Ap = npr.integers(-128, 128, (m, kt * 32), dtype=np.int64)
        Ap[:, :k] = A
        env.mem[cfg.RGN_ACT].write_np(a_off, Ap.reshape(-1), np.int8)
        # B tiles; padding garbage too
        Wp = npr.integers(-128, 128, (kt * 32, nt * 32), dtype=np.int64)
        Wp[:k, :n] = W
        stride = kt * cfg.TILE_BYTES + b_stride_pad
        for j in range(nt):
            for t in range(kt):
                env.mem[cfg.RGN_HBM].write_np(b_off + j * stride + t * 1024,
                                              Wp[t * 32:(t + 1) * 32, j * 32:(j + 1) * 32].reshape(-1), np.int8)
        Sp = npr.integers(-1000, 1000, nt * 32, dtype=np.int64)
        Sp[:n] = S
        env.mem[cfg.RGN_HBM].write_np(s_off, Sp, np.int32)
        f = dict(a_base=cfg.addr(cfg.RGN_ACT, a_off), b_base=cfg.addr(cfg.RGN_HBM, b_off // cfg.N_HBM),
                 s_base=cfg.addr(cfg.RGN_HBM, s_off), c_base=cfg.addr(cfg.RGN_SCRATCH, c_off),
                 m=m, k=k, n=n, b_is_kv=0, b_row_stride=stride // cfg.N_HBM if b_stride_pad else 0, out_fmt=fmt,
                 tag=(tag if tag is not None else i) & 0xF)
        exp = golden.gemm(A, W, S, fmt)
        label = f"gemm{i} m={m} k={k} n={n} fmt={fmt} stride_pad={b_stride_pad}"

        def check(it, f=f, exp=exp):
            _, nt_, tob = golden.mm_tile_geometry(f)
            eb, dt = golden.out_elem(fmt)
            errs = []
            for r in range(m):
                for j in range(nt_):
                    raw = env.mem[cfg.RGN_SCRATCH].read_np(c_off + r * nt_ * tob + j * tob, tob // eb, dt)
                    got = raw[:32].astype(np.int64)
                    want = np.zeros(32, dtype=np.int64)
                    cols = min(32, n - j * 32)
                    want[:cols] = exp[r, j * 32:j * 32 + cols]
                    bad = np.nonzero(got != want)[0]
                    for b in bad[:4]:
                        errs.append(f"C[{r},{j * 32 + b}] got {got[b]} want {want[b]}")
                    if np.any(raw[32:] != 0):
                        errs.append(f"row {r} tile {j}: nonzero beat padding")
            self.gemm_prof.append(dict(label=label, **Profile.gemm_metrics(f, it.acc_cycle, it.done_cycle)))
            return errs
        return CmdItem("mm", cfg.MM_CMD.pack(**f), f, check, label)

    async def body(self):
        items = [self.make_case(i, *c) for i, c in enumerate(self.cases)]
        await self.play("mm", items)

    def extra_profile(self):
        for g in self.gemm_prof:
            self.profile.add(f"gemm:{g.pop('label')}", **g)


@pyuvm.test()
class T1GemvFormats(T1Base):
    """M=1 decode GEMV: aligned + ragged K/N edges, all three output formats."""
    cases = [(1, 64, 64, cfg.OUT_VEC), (1, 70, 45, cfg.OUT_VEC), (1, 96, 33, cfg.OUT_I8),
             (1, 33, 100, cfg.OUT_ACC32), (1, 256, 128, cfg.OUT_I8), (1, 1, 1, cfg.OUT_VEC)]


@pyuvm.test()
class T1GemmPrefill(T1Base):
    """M>1 (prefill path: strided A per K-slice, B tile retained across M rows)."""
    cases = [(3, 64, 64, cfg.OUT_VEC), (5, 50, 70, cfg.OUT_I8), (2, 128, 32, cfg.OUT_ACC32)]


@pyuvm.test()
class T1StridedB(T1Base):
    """Nonzero b_row_stride (V-cache style physical stride between N tiles)."""
    cases = [(1, 64, 64, cfg.OUT_I8, 4096), (1, 40, 96, cfg.OUT_VEC, 2048)]


@pyuvm.test()
class T1BackToBackTags(T1Base):
    """8 random commands queued back-to-back, distinct tags; done_tag must follow each cmd."""

    async def body(self):
        rng = self.env.rng
        items = []
        for i in range(8):
            m = rng.choice([1, 1, 1, 2])
            items.append(self.make_case(i, m, rng.randint(1, 130), rng.randint(1, 130),
                                        rng.choice([cfg.OUT_I8, cfg.OUT_VEC, cfg.OUT_ACC32]), tag=15 - i))
        await self.play("mm", items)


@pyuvm.test()
class T1ProfileGemv(T1Base):
    """Profiling: zero-latency, no-backpressure memories -> intrinsic engine efficiency."""
    env_cls = T1FastEnv
    cases = [(1, 1024, 256, cfg.OUT_VEC), (1, 2048, 64, cfg.OUT_VEC), (4, 256, 128, cfg.OUT_VEC)]
