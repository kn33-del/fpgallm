"""T4 mem_subsys + ctrl_regs: act_buf, narrow HBM path, weight streamer (contract layout + BW
profiling), KV-cache scatter (contract), DMA, AXI-Lite registers, concurrency, AXI error."""
import random

import cocotb
import numpy as np
import pyuvm

from llmtb import cfg
from llmtb.bfm import AxiHbm, AxiLiteMaster, CmdDriver, Comp, MemRdClient, MemWrClient, rv
from llmtb.mem import Sparse, beat_to_bytes, bytes_to_beat
from llmtb.uvm import BaseEnv, BaseTest

RD = ["ifetch", "mm_a", "mm_s", "attn_rd", "vec_rd"]
WR = ["mm_c", "attn_wr", "kv", "vec_wr"]
PC_MASK = (1 << 29) - 1


class SeqSide(Comp):
    """Plays llm_ctrl's side of ctrl_regs: pulses ap_done when told, drives result_token."""

    def __init__(self, dut):
        super().__init__("seq_side")
        self.d = dut
        self.starts = 0
        self.perf_clr_seen = 0
        self.pulse = False
        self.token = 0

    def reset(self):
        self.d.ap_done.value = 0
        self.d.result_token.value = 0

    def drive(self):
        self.d.ap_done.value = int(self.pulse)
        self.d.result_token.value = self.token
        self.pulse = False

    def sample(self):
        if rv(self.d.ap_start):
            self.starts += 1


class T4Env(BaseEnv):
    hbm_kw = dict(lat=(20, 40), p_r=0.9)
    rd_kw = dict()
    wr_kw = dict()

    def make_bfms(self):
        d = self.dut
        mk = lambda n: random.Random(hash((self.seed, n)))
        self.rng = random.Random(self.seed)
        self.hbm = self.domain.add(AxiHbm(d, rng=mk("hbm"), **self.hbm_kw))
        self.rdc = {p: self.domain.add(MemRdClient(d, p, cfg.BEAT_W, rng=mk(p), **self.rd_kw)) for p in RD}
        self.wb = self.domain.add(MemRdClient(d, "mm_b", cfg.WSTREAM_W, rng=mk("mm_b"), name="mm_b", **self.rd_kw))
        self.wrc = {p: self.domain.add(MemWrClient(d, p, cfg.BEAT_W, rng=mk(p), **self.wr_kw)) for p in WR}
        self.dma = self.domain.add(CmdDriver(d, d.dma_cmd, d.dma_cmd_valid, d.dma_cmd_ready, d.dma_done,
                                             rng=mk("dma"), name="dma"))
        self.axil = self.domain.add(AxiLiteMaster(d))
        self.seq = self.domain.add(SeqSide(d))


class T4FastEnv(T4Env):
    hbm_kw = dict(lat=(30, 30), p_r=1.0, p_ar=1.0)
    rd_kw = dict(p_req=1.0, p_ready=1.0)
    wr_kw = dict(p_req=1.0, p_valid=1.0)


def rand_beats(rng, n):
    return [rng.getrandbits(256) for _ in range(n)]


class T4Base(BaseTest):
    env_cls = T4Env

    def build_phase(self):
        super().build_phase()
        self.errs = []

    def extra_checks(self):
        return self.errs

    def expect(self, cond, msg):
        if not cond:
            self.errs.append(msg)

    async def wr(self, port, beats, rgn, off, row_beats, n_rows=1, stride=0):
        ev = self.env.wrc[port].write(beats, base=cfg.addr(rgn, off), row_beats=row_beats, n_rows=n_rows,
                                      row_stride=stride)
        return await ev.wait()

    async def rd(self, port, rgn, off, row_beats, n_rows=1, stride=0):
        cli = self.env.wb if port == "mm_b" else self.env.rdc[port]
        ev = cli.read(base=cfg.addr(rgn, off), row_beats=row_beats, n_rows=n_rows, row_stride=stride)
        return await ev.wait()


@pyuvm.test()
class T4ActBuf(T4Base):
    """ACT + SCRATCH: linear and 2D-strided writes, read back through two different clients."""

    async def body(self):
        rng = self.env.rng
        for rgn, port_w, port_r in ((cfg.RGN_ACT, "vec_wr", "vec_rd"), (cfg.RGN_SCRATCH, "mm_c", "mm_a"),
                                    (cfg.RGN_SCRATCH, "attn_wr", "attn_rd")):
            b = rand_beats(rng, 24)
            await self.wr(port_w, b, rgn, 0x400, 24)
            got = (await self.rd(port_r, rgn, 0x400, 24))["beats"]
            self.expect(got == b, f"rgn {rgn}: linear readback mismatch via {port_w}->{port_r}")
            b2 = rand_beats(rng, 12)                         # 4 rows x 3 beats, stride 8 beats
            await self.wr(port_w, b2, rgn, 0x2000, 3, 4, 256)
            got = (await self.rd(port_r, rgn, 0x2000, 3, 4, 256))["beats"]
            self.expect(got == b2, f"rgn {rgn}: 2D readback mismatch")
            got = (await self.rd(port_r, rgn, 0x2000 + 256, 3))["beats"]
            self.expect(got == b2[3:6], f"rgn {rgn}: row 1 not at base+stride")


@pyuvm.test()
class T4NarrowHbm(T4Base):
    """Narrow HBM writes/reads: several PCs, 4 KB-boundary-crossing bursts, 2D, >64-beat rows."""

    async def body(self):
        rng, hbm = self.env.rng, self.env.hbm
        cases = [(3, 0x100, 8, 1, 0), (17, 0xF80, 10, 1, 0),        # crosses 4 KB
                 (31, 0x20000, 100, 1, 0), (0, 0x3000, 2, 5, 0x400)]  # 100 beats > NB; 2D
        for pc, off, rb, nr, st in cases:
            addr = (pc << 29) | off
            b = rand_beats(rng, rb * nr)
            await self.wr("attn_wr", b, cfg.RGN_HBM, addr, rb, nr, st)
            ok = True
            for r in range(nr):
                for c in range(rb):
                    if bytes_to_beat(hbm.mem.read(addr + r * st + c * 32, 32)) != b[r * rb + c]:
                        ok = False
            self.expect(ok, f"HBM contents wrong after write pc={pc} off={off:#x}")
            got = (await self.rd("mm_s", cfg.RGN_HBM, addr, rb, nr, st))["beats"]
            self.expect(got == b, f"HBM readback mismatch pc={pc} off={off:#x} ({rb}x{nr})")


def put_tiles(mem, pc_off, tiles):
    """Contract weight layout: tile t row r -> PC r, per-PC offset pc_off + t*32."""
    for t, tile in enumerate(tiles):
        for r in range(32):
            mem.write((r << 29) | (pc_off + t * 32), tile[r].tobytes())


def beat_to_tile(beat):
    return np.frombuffer(beat_to_bytes(beat, 1024), np.int8).reshape(32, 32)


@pyuvm.test()
class T4WeightStream(T4Base):
    """Weight streamer: per-PC addressing per llm_pkg contract, 2D rows, backpressure."""

    async def body(self):
        g = np.random.default_rng(1)
        kt, nt = 5, 3
        tiles = [g.integers(-128, 128, (32, 32), dtype=np.int8) for _ in range(kt * nt)]
        base = 0x0040_0000
        put_tiles(self.env.hbm.mem, base, tiles)
        got = (await self.rd("mm_b", cfg.RGN_HBM, base, kt, nt, kt * 32))["beats"]
        self.expect(len(got) == kt * nt, f"{len(got)} wide beats")
        for t, b in enumerate(got):
            self.expect(np.array_equal(beat_to_tile(b), tiles[t]), f"tile {t} mismatch")
        # partial rows (K^T-style prefix read): first 2 k-tiles of each n row
        got = (await self.rd("mm_b", cfg.RGN_HBM, base, 2, nt, kt * 32))["beats"]
        for j in range(nt):
            for i in range(2):
                self.expect(np.array_equal(beat_to_tile(got[j * 2 + i]), tiles[j * kt + i]),
                            f"prefix read n={j} k={i} mismatch")


@pyuvm.test()
class T4WeightStreamBW(T4Base):
    """Profiling: 512-tile stream, fixed 30-cycle HBM latency, no backpressure."""
    env_cls = T4FastEnv

    async def body(self):
        kt, nt = 64, 8
        base = 0x0100_0000
        await self.env.axil.write(0x000, 1)                 # ap_start -> perf_clr
        r = await self.rd("mm_b", cfg.RGN_HBM, base, kt, nt, kt * 32)
        n = len(r["beats"])
        span = r["last_cycle"] - r["req_cycle"]
        await self.env.domain.cycles(2)                     # last handshake commits at the next edge
        perf = cfg.PERF.unpack(rv(self.env.dut.perf))
        gbps = n * 1024 / span * cfg.KERNEL_MHZ * 1e6 / 1e9
        self.profile.add("weight_stream", tiles=n, cycles=span, first_beat_latency=r["first_cycle"] - r["req_cycle"],
                         tiles_per_cycle=round(n / span, 4), GBps_at_300MHz=round(gbps, 1),
                         pct_port_ceiling=round(100 * gbps / cfg.PORT_CEIL_GBPS, 1), **perf)
        self.expect(perf["ws_beats"] == n, f"perf.ws_beats={perf['ws_beats']} but {n} tiles delivered")
        self.expect(n / span > 0.85, f"streamer delivered only {n / span:.2f} tiles/cycle with an ideal HBM")


@pyuvm.test()
class T4KvCache(T4Base):
    """kv_wr (kv_tag_t contract) -> K^T column + V row scatter; read back through the streamer."""

    async def body(self):
        g = np.random.default_rng(2)
        layer, kvh = 3, 1
        positions = [0, 1, 31, 32, 45]
        K = {p: g.integers(-128, 128, cfg.HEAD_DIM, dtype=np.int8) for p in positions}
        Vv = {p: g.integers(-128, 128, cfg.HEAD_DIM, dtype=np.int8) for p in positions}
        for p in positions:
            for is_v, vec in ((0, K[p]), (1, Vv[p])):
                tag = cfg.KV_TAG.pack(layer=layer, kvh=kvh, is_v=is_v, pos=p)
                b = vec.tobytes()
                await self.wr("kv", [bytes_to_beat(b[:32]), bytes_to_beat(b[32:])], cfg.RGN_HBM, tag, 2)
        pmax = max(positions)
        ntp = cfg.ceil_div(pmax + 1, 32)
        kb, vb = cfg.kv_pc_off(layer, kvh, 0), cfg.kv_pc_off(layer, kvh, 1)
        kt_beats = (await self.rd("mm_b", cfg.RGN_HBM, kb, 2 * ntp, 1))["beats"]
        v_beats = (await self.rd("mm_b", cfg.RGN_HBM, vb, ntp, 2, (cfg.MAX_SEQ // 32) * 32))["beats"]
        for p in positions:
            for d in range(cfg.HEAD_DIM):
                t = beat_to_tile(kt_beats[(p // 32) * 2 + d // 32])        # tile(k=d/32, n=p/32)
                if t[d % 32][p % 32] != K[p][d]:
                    self.errs.append(f"K^T pos={p} dim={d}: got {t[d % 32][p % 32]} want {K[p][d]}")
                    break
                tv = beat_to_tile(v_beats[(d // 32) * ntp + p // 32])      # tile(k=p/32, n=d/32)
                if tv[p % 32][d % 32] != Vv[p][d]:
                    self.errs.append(f"V pos={p} dim={d}: got {tv[p % 32][d % 32]} want {Vv[p][d]}")
                    break


@pyuvm.test()
class T4Dma(T4Base):
    """OP_EMBED-style copy: one embedding row (D_MODEL int16 = 128 beats) HBM -> ACT."""

    async def body(self):
        rng = self.env.rng
        n = cfg.D_MODEL * 2 // 32
        src = (5 << 29) | 0x8000
        b = rand_beats(rng, n)
        for i, x in enumerate(b):
            self.env.hbm.mem.write(src + 32 * i, beat_to_bytes(x, 32))
        ev = self.env.dma.send(cfg.DMA_CMD.pack(src=cfg.addr(cfg.RGN_HBM, src), dst=cfg.addr(cfg.RGN_ACT, 0x1000),
                                                len_beats=n))
        await ev.wait()
        got = (await self.rd("vec_rd", cfg.RGN_ACT, 0x1000, n))["beats"]
        self.expect(got == b, "DMA HBM->ACT data mismatch")
        ev = self.env.dma.send(cfg.DMA_CMD.pack(src=cfg.addr(cfg.RGN_ACT, 0x1000),
                                                dst=cfg.addr(cfg.RGN_SCRATCH, 0x40), len_beats=7))
        await ev.wait()
        got = (await self.rd("vec_rd", cfg.RGN_SCRATCH, 0x40, 7))["beats"]
        self.expect(got == b[:7], "DMA ACT->SCRATCH data mismatch")


@pyuvm.test()
class T4CtrlRegs(T4Base):
    """AXI-Lite: RW regs, ap_start pulse, done/idle/ready bits, COR, interrupt (GIE/IER/ISR), cycles."""

    async def body(self):
        ax, d, seq = self.env.axil, self.env.dut, self.env.seq
        regs = {0x10: 0x1234_5000, 0x14: 0x2, 0x18: 77, 0x1C: 41, 0x24: 0xBEEF}
        for a, v in regs.items():
            await ax.write(a, v)
        for a, v in regs.items():
            got = await ax.read(a)
            self.expect(got == v, f"reg {a:#x} read {got:#x} want {v:#x}")
        rgn, off = cfg.addr_split(rv(d.instr_base))
        self.expect(off == (0x2 << 32 | 0x1234_5000) and rgn == cfg.RGN_HBM, f"instr_base {off:#x}")
        self.expect(rv(d.n_instr) == 77 and rv(d.token_pos) == 41 and rv(d.token_in) == 0xBEEF, "decoded regs")
        await ax.write(0x10, 0xAB, strb=0x1)                 # byte strobe merge
        self.expect(await ax.read(0x10) == 0x1234_50AB, "wstrb merge")
        self.expect((await ax.read(0x00)) & 0x4, "ap_idle not set at reset")
        await ax.write(0x04, 1)
        await ax.write(0x08, 1)
        await ax.write(0x00, 1)
        self.expect(seq.starts == 1, f"{seq.starts} ap_start pulses")
        st = await ax.read(0x00)
        self.expect(st & 1 and not st & 4, f"ap_ctrl while busy = {st:#x}")
        await ax.write(0x00, 1)                              # start while busy: ignored
        self.expect(seq.starts == 1, "second start while busy was not ignored")
        await self.env.domain.cycles(50)
        seq.token = 0x1F2E
        seq.pulse = True
        await self.env.domain.cycles(3)
        self.expect(rv(d.interrupt) == 1, "interrupt not raised on done with GIE+IER[0]")
        st = await ax.read(0x00)
        self.expect(st & 0x2 and st & 0x4 and st & 0x8, f"done/idle/ready not set: {st:#x}")
        st = await ax.read(0x00)
        self.expect(not st & 0x2, "ap_done not clear-on-read")
        self.expect(await ax.read(0x20) == 0x1F2E, "result_token")
        cyc = await ax.read(0x40)
        self.expect(50 <= cyc <= 200, f"cycle counter {cyc}")
        await ax.write(0x0C, 1)                              # ISR toggle-on-write clears
        await self.env.domain.cycles(2)
        self.expect(rv(d.interrupt) == 0, "ISR toggle did not clear interrupt")
        self.expect(await ax.read(0x7FC) == 0xDEADBEEF, "unmapped read")


@pyuvm.test()
class T4Concurrent(T4Base):
    """Everyone at once: weight stream + 3 narrow readers + 3 writers (ACT/SCRATCH/HBM) + DMA."""

    async def body(self):
        rng = self.env.rng
        g = np.random.default_rng(3)
        tiles = [g.integers(-128, 128, (32, 32), dtype=np.int8) for _ in range(40)]
        put_tiles(self.env.hbm.mem, 0x0080_0000, tiles)
        pre = {}
        for i, (rgn, off) in enumerate(((cfg.RGN_HBM, (9 << 29) | 0x1000), (cfg.RGN_ACT, 0x8000),
                                        (cfg.RGN_SCRATCH, 0x9000))):
            pre[i] = (rgn, off, rand_beats(rng, 16))
            await self.wr("vec_wr", pre[i][2], rgn, off, 16)

        async def stream():
            got = (await self.rd("mm_b", cfg.RGN_HBM, 0x0080_0000, 8, 5, 8 * 32))["beats"]
            for t, b in enumerate(got):
                self.expect(np.array_equal(beat_to_tile(b), tiles[t]), f"concurrent stream tile {t}")

        async def reader(port, i):
            rgn, off, b = pre[i]
            for _ in range(3):
                got = (await self.rd(port, rgn, off, 16))["beats"]
                self.expect(got == b, f"{port} concurrent readback of region {i}")

        async def writer(port, rgn, off):
            b = rand_beats(rng, 20)
            await self.wr(port, b, rgn, off, 20)
            got = (await self.rd("ifetch", rgn, off, 20))["beats"]
            self.expect(got == b, f"{port} concurrent write rgn={rgn}")

        tasks = [cocotb.start_soon(c) for c in (
            stream(), reader("mm_a", 0), reader("attn_rd", 1), reader("mm_s", 2),
            writer("mm_c", cfg.RGN_SCRATCH, 0xA000), writer("attn_wr", cfg.RGN_HBM, (12 << 29) | 0x40),
            writer("vec_wr", cfg.RGN_ACT, 0xC000))]
        for t in tasks:
            await t


@pyuvm.test()
class T4AxiError(T4Base):
    """Non-OKAY rresp sets the sticky axi_err; the next ap_start clears it."""

    async def body(self):
        self.env.hbm.resp_err[4] = 2                     # SLVERR on port 4
        await self.rd("mm_s", cfg.RGN_HBM, (4 << 29) | 0x100, 2)
        self.env.hbm.resp_err.clear()
        await self.env.domain.cycles(2)
        self.expect(await self.env.axil.read(0x68) == 1, "axi_err not set after SLVERR")
        await self.env.axil.write(0x00, 1)
        await self.env.domain.cycles(2)
        self.expect(await self.env.axil.read(0x68) == 0, "axi_err not cleared by ap_start")
