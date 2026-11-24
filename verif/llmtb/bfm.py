"""Cycle-level bus functional models.

Timing discipline (every BFM follows it, which keeps handshakes race-free):
  falling edge   : every component drive()s its outputs for the coming rising edge
  falling + 1 ns : every component sample()s; a transfer is recorded iff valid&&ready
                   *as they will be seen* by the next rising edge.
A single ClockDomain loop calls all components so the ordering is global.
"""
import random
from collections import deque

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Event, FallingEdge, RisingEdge, Timer

from . import cfg
from .mem import Sparse, beat_to_bytes, bytes_to_beat


def rv(h):
    v = h.value
    try:
        return v.to_unsigned()
    except AttributeError:
        return int(v)


class Pending:
    """Completion handle: `result = await handle.wait()`."""

    def __init__(self):
        self.ev = Event()
        self.result = None

    def done(self, result):
        self.result = result
        self.ev.set()

    async def wait(self):
        await self.ev.wait()
        return self.result


class Stats(dict):
    def inc(self, k, n=1):
        self[k] = self.get(k, 0) + n


# =========================================================================================
class ClockDomain:
    def __init__(self, dut, clk_name="clk", period_ns=10):
        self.dut = dut
        self.clk = getattr(dut, clk_name)
        self.period_ns = period_ns
        self.comps = []
        self.cycle = 0
        self._task = None

    def add(self, comp):
        self.comps.append(comp)
        comp.domain = self
        return comp

    async def start(self, rst_name="rst_n", reset_cycles=8):
        cocotb.start_soon(Clock(self.clk, self.period_ns, unit="ns").start())
        rst = getattr(self.dut, rst_name)
        rst.value = 0
        for c in self.comps:
            c.reset()
        for _ in range(reset_cycles):
            await FallingEdge(self.clk)
        self._task = cocotb.start_soon(self._loop())
        await FallingEdge(self.clk)
        rst.value = 1
        await RisingEdge(self.clk)

    async def _loop(self):
        while True:
            await FallingEdge(self.clk)
            self.cycle += 1
            for c in self.comps:
                c.drive()
            await Timer(1, unit="ns")
            for c in self.comps:
                c.sample()

    async def cycles(self, n):
        for _ in range(n):
            await FallingEdge(self.clk)

    async def wait_until(self, pred, timeout=2_000_000, what="condition"):
        start = self.cycle
        while not pred():
            await FallingEdge(self.clk)
            if self.cycle - start > timeout:
                raise TimeoutError(f"timeout after {timeout} cycles waiting for {what}")


class Comp:
    domain: ClockDomain = None

    def __init__(self, name, rng=None):
        self.name = name
        self.rng = rng or random.Random(0)
        self.stats = Stats()
        self.errors = []

    def reset(self):
        pass

    def drive(self):
        pass

    def sample(self):
        pass

    def coin(self, p):
        return p >= 1.0 or self.rng.random() < p

    def err(self, msg):
        self.errors.append(f"[{self.name} @cyc {self.domain.cycle if self.domain else -1}] {msg}")


class Port:
    """Bundle of <prefix>_<sig> handles on the wrapper toplevel."""

    def __init__(self, dut, prefix, sigs):
        for s in sigs:
            setattr(self, s, getattr(dut, f"{prefix}_{s}"))


# =========================================================================================
# Memory servers (DUT is client)
# =========================================================================================
def logical_beats(mem, req, width_bits):
    """Default address map for unit tests: beat (row, col) at base.off + row*stride + col*W/8."""
    nb = width_bits // 8
    rgn, off = cfg.addr_split(req["base"])
    out = []
    for r in range(req["n_rows"]):
        for c in range(req["row_beats"]):
            out.append(bytes_to_beat(mem[rgn].read(off + r * req["row_stride"] + c * nb, nb)))
    return out


def wide_beats(mem, req):
    """Wide (8192-b) B port, per-PC addressing (llm_pkg contract): beat (row, col) lives at per-PC
    offset off[28:0] + row*stride + col*32; the unit benches store tiles contiguously at
    logical byte = per-PC byte * N_HBM."""
    rgn, off = cfg.addr_split(req["base"])
    off &= cfg.PC_MASK
    out = []
    for r in range(req["n_rows"]):
        for c in range(req["row_beats"]):
            pc_off = off + r * req["row_stride"] + c * cfg.TILE_PC_BYTES
            out.append(bytes_to_beat(mem[rgn].read(pc_off * cfg.N_HBM, cfg.TILE_BYTES)))
    return out


def logical_addr(req, idx, width_bits):
    nb = width_bits // 8
    rgn, off = cfg.addr_split(req["base"])
    r, c = divmod(idx, req["row_beats"])
    return rgn, off + r * req["row_stride"] + c * nb


class MemRdServer(Comp):
    def __init__(self, dut, prefix, width, beats_fn, rng=None, p_req=0.7, p_valid=0.8, lat=(1, 6),
                 name=None):
        super().__init__(name or prefix, rng)
        self.p = Port(dut, prefix, ["req", "req_valid", "req_ready", "data", "valid", "ready", "last"])
        self.width = width
        self.beats_fn = beats_fn
        self.p_req, self.p_valid, self.lat = p_req, p_valid, lat
        self.reqs = []
        self.on_req = []

    def reset(self):
        self.q = deque()
        self.busy = False
        self.lat_left = 0
        self._rr = self._v = 0
        self._last_data = None
        self.p.req_ready.value = 0
        self.p.valid.value = 0
        self.p.last.value = 0

    def drive(self):
        self._rr = int((not self.busy) and self.coin(self.p_req))
        self.p.req_ready.value = self._rr
        self._v = int(bool(self.q) and self.lat_left == 0 and self.coin(self.p_valid))
        self.p.valid.value = self._v
        if self.q:
            d, last = self.q[0]
            if d is not self._last_data:
                self.p.data.value = d
                self._last_data = d
            self.p.last.value = int(last)
        else:
            self.p.last.value = 0

    def sample(self):
        if self.busy:
            self.stats.inc("busy_cycles")
        if self._v:
            if rv(self.p.ready):
                self.q.popleft()
                self.stats.inc("beats")
                if not self.q:
                    self.busy = False
                    self.stats.inc("done_reqs")
            else:
                self.stats.inc("dut_stall")
        elif self.busy and self.q and self.lat_left == 0:
            self.stats.inc("tb_gap")
        if self.lat_left:
            self.lat_left -= 1
        if self._rr and rv(self.p.req_valid):
            req = cfg.MEM_REQ.unpack(rv(self.p.req))
            req["cycle"] = self.domain.cycle
            self.reqs.append(req)
            for cb in self.on_req:
                cb(req)
            beats = self.beats_fn(req)
            n = len(beats)
            self.q = deque((b, i == n - 1) for i, b in enumerate(beats))
            self.busy = n > 0
            self.lat_left = self.rng.randint(*self.lat)
            self.stats.inc("reqs")


class MemWrServer(Comp):
    def __init__(self, dut, prefix, width, beat_fn, rng=None, p_req=0.7, p_ready=0.8, done_lat=(1, 6),
                 name=None):
        super().__init__(name or prefix, rng)
        self.p = Port(dut, prefix, ["req", "req_valid", "req_ready", "data", "valid", "last", "ready", "done"])
        self.width = width
        self.beat_fn = beat_fn            # beat_fn(req, idx, data_int)
        self.p_req, self.p_ready, self.done_lat = p_req, p_ready, done_lat
        self.reqs = []
        self.on_complete = []             # cb(req, [beats])

    def reset(self):
        self.busy = False
        self.left = 0
        self.idx = 0
        self.cur = None
        self.beats = []
        self.done_cd = -1
        self._rr = self._r = self._done = 0
        for s in (self.p.req_ready, self.p.ready, self.p.done):
            s.value = 0

    def drive(self):
        self._done = int(self.done_cd == 0)
        self.p.done.value = self._done
        self._rr = int((not self.busy) and self.coin(self.p_req))
        self.p.req_ready.value = self._rr
        self._r = int(self.busy and self.left > 0 and self.coin(self.p_ready))
        self.p.ready.value = self._r

    def sample(self):
        if self.busy:
            self.stats.inc("busy_cycles")
        if self._done:
            self.done_cd = -1
            self.busy = False
            for cb in self.on_complete:
                cb(self.cur, self.beats)
        elif self.done_cd > 0:
            self.done_cd -= 1
        if self._r:
            if rv(self.p.valid):
                d = rv(self.p.data)
                last = rv(self.p.last)
                if bool(last) != (self.left == 1):
                    self.err(f"last={last} on beat {self.idx} of {self.idx + self.left}")
                self.beat_fn(self.cur, self.idx, d)
                self.beats.append(d)
                self.idx += 1
                self.left -= 1
                self.stats.inc("beats")
                if self.left == 0:
                    self.done_cd = self.rng.randint(*self.done_lat)
            else:
                self.stats.inc("tb_wait_data")
        if self._rr and rv(self.p.req_valid):
            req = cfg.MEM_REQ.unpack(rv(self.p.req))
            req["cycle"] = self.domain.cycle
            self.reqs.append(req)
            self.cur = req
            self.beats = []
            self.idx = 0
            self.left = req["row_beats"] * req["n_rows"]
            self.busy = True
            self.stats.inc("reqs")
            if self.left == 0:
                self.done_cd = self.rng.randint(*self.done_lat)


# =========================================================================================
# Memory clients (DUT is server) -- used by the T4 bench
# =========================================================================================
class MemRdClient(Comp):
    def __init__(self, dut, prefix, width, rng=None, p_req=0.8, p_ready=0.8, name=None):
        super().__init__(name or prefix, rng)
        self.p = Port(dut, prefix, ["req", "req_valid", "req_ready", "data", "valid", "ready", "last"])
        self.width, self.p_req, self.p_ready = width, p_req, p_ready
        self.ops = deque()

    def reset(self):
        self.cur = None
        self._rqv = self._r = 0
        self.p.req_valid.value = 0
        self.p.ready.value = 0

    def read(self, **req):
        """Returns an Event; event.data -> {'beats': [...], 'req_cycle', 'first_cycle', 'last_cycle'}."""
        ev = Pending()
        self.ops.append({"req": cfg.MEM_REQ.pack(**req), "n": req["row_beats"] * req["n_rows"], "beats": [],
                         "ev": ev, "sent": False})
        return ev

    def drive(self):
        if self.cur is None and self.ops:
            self.cur = self.ops.popleft()
        c = self.cur
        self._rqv = int(bool(c and not c["sent"]) and self.coin(self.p_req))
        self.p.req_valid.value = self._rqv
        if c and not c["sent"]:
            self.p.req.value = c["req"]
        self._r = int(bool(c and c["sent"]) and self.coin(self.p_ready))
        self.p.ready.value = self._r

    def sample(self):
        c = self.cur
        if not c:
            return
        self.stats.inc("busy_cycles")
        if self._rqv and rv(self.p.req_ready):
            c["sent"] = True
            c["req_cycle"] = self.domain.cycle
            self.stats.inc("reqs")
        elif c["sent"] and self._r and rv(self.p.valid):
            c["beats"].append(rv(self.p.data))
            c.setdefault("first_cycle", self.domain.cycle)
            self.stats.inc("beats")
            last = rv(self.p.last)
            if bool(last) != (len(c["beats"]) == c["n"]):
                self.err(f"last={last} at beat {len(c['beats'])}/{c['n']}")
            if len(c["beats"]) == c["n"] or last:
                c["last_cycle"] = self.domain.cycle
                c["ev"].done(c)
                self.cur = None
        elif c["sent"] and self._r:
            self.stats.inc("dut_starve")


class MemWrClient(Comp):
    def __init__(self, dut, prefix, width, rng=None, p_req=0.8, p_valid=0.8, name=None):
        super().__init__(name or prefix, rng)
        self.p = Port(dut, prefix, ["req", "req_valid", "req_ready", "data", "valid", "last", "ready", "done"])
        self.width, self.p_req, self.p_valid = width, p_req, p_valid
        self.ops = deque()

    def reset(self):
        self.cur = None
        self._rqv = self._v = 0
        self.p.req_valid.value = 0
        self.p.valid.value = 0
        self.p.last.value = 0

    def write(self, beats, **req):
        ev = Pending()
        assert len(beats) == req["row_beats"] * req["n_rows"]
        self.ops.append({"req": cfg.MEM_REQ.pack(**req), "beats": list(beats), "i": 0, "ev": ev,
                         "sent": False, "all": False})
        return ev

    def drive(self):
        if self.cur is None and self.ops:
            self.cur = self.ops.popleft()
        c = self.cur
        self._rqv = int(bool(c and not c["sent"]) and self.coin(self.p_req))
        self.p.req_valid.value = self._rqv
        if c and not c["sent"]:
            self.p.req.value = c["req"]
        self._v = int(bool(c and c["sent"] and not c["all"]) and self.coin(self.p_valid))
        self.p.valid.value = self._v
        if c and c["sent"] and not c["all"]:
            self.p.data.value = c["beats"][c["i"]]
            self.p.last.value = int(c["i"] == len(c["beats"]) - 1)

    def sample(self):
        c = self.cur
        if not c:
            return
        self.stats.inc("busy_cycles")
        if c["sent"] and rv(self.p.done):
            if not c["all"]:
                self.err("done before all beats were accepted")
            c["done_cycle"] = self.domain.cycle
            c["ev"].done(c)
            self.cur = None
            return
        if self._rqv and rv(self.p.req_ready):
            c["sent"] = True
            self.stats.inc("reqs")
            if not c["beats"]:
                c["all"] = True
        elif self._v and rv(self.p.ready):
            c["i"] += 1
            self.stats.inc("beats")
            if c["i"] == len(c["beats"]):
                c["all"] = True


# =========================================================================================
# Command channels
# =========================================================================================
class CmdDriver(Comp):
    """Generic valid/ready command port + an untagged done pulse (in-order completion)."""

    def __init__(self, dut, cmd, valid, ready, done, rng=None, p_valid=0.8, name="cmd", on_done=None):
        super().__init__(name, rng)
        self.cmd, self.valid, self.ready, self.done_s = cmd, valid, ready, done
        self.p_valid = p_valid
        self.on_done = on_done
        self.pending = deque()
        self.inflight = deque()
        self.log = []

    def reset(self):
        self._v = 0
        self.valid.value = 0

    def send(self, cmd_int, meta=None):
        ev = Event()
        self.pending.append({"cmd": cmd_int, "meta": meta, "ev": ev})
        return ev

    def drive(self):
        self._v = int(bool(self.pending) and self.coin(self.p_valid))
        self.valid.value = self._v
        if self.pending:
            self.cmd.value = self.pending[0]["cmd"]

    def sample(self):
        if self.inflight and rv(self.done_s):
            t = self.inflight.popleft()
            t["done_cycle"] = self.domain.cycle
            if self.on_done:
                self.on_done(t)
            t["ev"].set()
        if self._v and rv(self.ready):
            t = self.pending.popleft()
            t["acc_cycle"] = self.domain.cycle
            self.inflight.append(t)
            self.log.append(t)
            self.stats.inc("cmds")


class MmMaster(CmdDriver):
    """Drives mm_if into an engine; matches done by done_tag."""

    def __init__(self, dut, prefix, rng=None, p_valid=0.8, name="mm_master"):
        p = Port(dut, prefix, ["cmd", "cmd_valid", "cmd_ready", "done", "done_tag"])
        super().__init__(dut, p.cmd, p.cmd_valid, p.cmd_ready, p.done, rng, p_valid, name)
        self.done_tag = p.done_tag

    def sample(self):
        if self.inflight and rv(self.done_s):
            t = self.inflight.popleft()
            tag = rv(self.done_tag)
            exp = cfg.MM_CMD.unpack(t["cmd"])["tag"]
            if tag != exp:
                self.err(f"done_tag {tag} != cmd tag {exp}")
            t["done_cycle"] = self.domain.cycle
            t["ev"].set()
        if self._v and rv(self.ready):
            t = self.pending.popleft()
            t["acc_cycle"] = self.domain.cycle
            self.inflight.append(t)
            self.log.append(t)
            self.stats.inc("cmds")


class MmEngineModel(Comp):
    """Reference T1 engine behind a DUT's mm_if master port (for T2/T3 unit tests)."""

    def __init__(self, dut, prefix, mem, rng=None, p_ready=0.7, lat=(4, 30), execute=None,
                 name="mm_model"):
        super().__init__(name, rng)
        self.p = Port(dut, prefix, ["cmd", "cmd_valid", "cmd_ready", "done", "done_tag"])
        self.mem = mem
        self.p_ready, self.lat = p_ready, lat
        self.execute = execute
        self.cmds = []

    def reset(self):
        self.busy = False
        self.cd = -1
        self.tag = 0
        self._r = self._d = 0
        self.p.cmd_ready.value = 0
        self.p.done.value = 0

    def drive(self):
        self._d = int(self.cd == 0)
        self.p.done.value = self._d
        self.p.done_tag.value = self.tag
        self._r = int((not self.busy) and self.coin(self.p_ready))
        self.p.cmd_ready.value = self._r

    def sample(self):
        if self.busy:
            self.stats.inc("busy_cycles")
        if self._d:
            self.busy = False
            self.cd = -1
        elif self.cd > 0:
            self.cd -= 1
        if self._r and rv(self.p.cmd_valid):
            c = cfg.MM_CMD.unpack(rv(self.p.cmd))
            c["cycle"] = self.domain.cycle
            self.cmds.append(c)
            self.tag = c["tag"]
            if self.execute:
                self.execute(c)
            self.busy = True
            self.cd = self.rng.randint(*self.lat)
            self.stats.inc("cmds")


# =========================================================================================
# 32-port HBM AXI slave
# =========================================================================================
class AxiHbm(Comp):
    """N_HBM AXI4 slave ports onto one flat 34-bit byte memory (HBM global address map).

    lat: AR->first R beat latency (cycles). p_r: per-cycle R-valid probability per port.
    """

    def __init__(self, dut, mem=None, rng=None, lat=(20, 40), p_ar=0.9, p_r=1.0, p_aw=0.9, p_w=0.9,
                 b_lat=(4, 12), max_ar=16, name="hbm"):
        super().__init__(name, rng)
        self.d = dut
        self.mem = mem if mem is not None else Sparse()
        self.lat, self.p_ar, self.p_r, self.p_aw, self.p_w, self.b_lat = lat, p_ar, p_r, p_aw, p_w, b_lat
        self.max_ar = max_ar
        self.N = cfg.N_HBM
        self.port_stats = [Stats() for _ in range(self.N)]
        self.resp_err = {}            # (port) -> forced rresp (for error injection)

    def reset(self):
        N = self.N
        self.ar = [deque() for _ in range(N)]      # [addr, len, id, ready_cycle, beat]
        self.aw = [deque() for _ in range(N)]
        self.wbeat = [0] * N
        self.b = [deque() for _ in range(N)]       # [id, ready_cycle]
        self._arr = self._rv = self._awr = self._wr = self._bv = 0
        for s in ("m_axi_arready", "m_axi_rvalid", "m_axi_awready", "m_axi_wready", "m_axi_bvalid",
                  "m_axi_rlast", "m_axi_rresp", "m_axi_bresp"):
            getattr(self.d, s).value = 0

    def drive(self):
        N, cyc = self.N, self.domain.cycle
        arr = rvld = rlast = awr = wr = bv = 0
        rdata = rid = bid = rresp = 0
        for p in range(N):
            if len(self.ar[p]) < self.max_ar and self.coin(self.p_ar):
                arr |= 1 << p
            if self.ar[p] and self.ar[p][0][3] <= cyc and self.coin(self.p_r):
                a = self.ar[p][0]
                rvld |= 1 << p
                beat_addr = a[0] + a[4] * 32
                rdata |= bytes_to_beat(self.mem.read(beat_addr, 32)) << (256 * p)
                rid |= a[2] << (cfg.AXI_IDW * p)
                rresp |= self.resp_err.get(p, 0) << (2 * p)
                if a[4] == a[1]:
                    rlast |= 1 << p
            if self.coin(self.p_aw):
                awr |= 1 << p
            if self.aw[p] and self.coin(self.p_w):
                wr |= 1 << p
            if self.b[p] and self.b[p][0][1] <= cyc:
                bv |= 1 << p
                bid |= self.b[p][0][0] << (cfg.AXI_IDW * p)
        self._arr, self._rv, self._awr, self._wr, self._bv = arr, rvld, awr, wr, bv
        d = self.d
        d.m_axi_arready.value = arr
        d.m_axi_rvalid.value = rvld
        d.m_axi_rlast.value = rlast
        d.m_axi_rresp.value = rresp
        if rvld:
            d.m_axi_rdata.value = rdata
            d.m_axi_rid.value = rid
        d.m_axi_awready.value = awr
        d.m_axi_wready.value = wr
        d.m_axi_bvalid.value = bv
        d.m_axi_bid.value = bid

    def sample(self):
        d, N, cyc = self.d, self.N, self.domain.cycle
        AW, IDW = cfg.M_AXI_AW, cfg.AXI_IDW
        if self._rv:
            rr = rv(d.m_axi_rready) & self._rv
            for p in range(N):
                if (self._rv >> p) & 1:
                    if (rr >> p) & 1:
                        a = self.ar[p][0]
                        self.port_stats[p].inc("r_beats")
                        a[4] += 1
                        if a[4] > a[1]:
                            self.ar[p].popleft()
                    else:
                        self.port_stats[p].inc("r_stall")
        arv = rv(d.m_axi_arvalid) & self._arr
        if arv:
            araddr, arlen, arid = rv(d.m_axi_araddr), rv(d.m_axi_arlen), rv(d.m_axi_arid)
            for p in range(N):
                if (arv >> p) & 1:
                    a = (araddr >> (AW * p)) & ((1 << AW) - 1)
                    ln = (arlen >> (8 * p)) & 0xFF
                    if (a & 0xFFF) + (ln + 1) * 32 > 4096:
                        self.err(f"port {p} AR crosses 4KB: addr={a:#x} len={ln + 1}")
                    if a & 31:
                        self.err(f"port {p} AR unaligned {a:#x}")
                    self.ar[p].append([a, ln, (arid >> (IDW * p)) & ((1 << IDW) - 1),
                                       cyc + self.rng.randint(*self.lat), 0])
                    self.port_stats[p].inc("ar")
                    self.port_stats[p].inc("ar_beats", ln + 1)
                    if (a >> 29) != p:
                        self.port_stats[p].inc("ar_cross_pc")
        awv = rv(d.m_axi_awvalid) & self._awr
        if awv:
            awaddr, awlen, awid = rv(d.m_axi_awaddr), rv(d.m_axi_awlen), rv(d.m_axi_awid)
            for p in range(N):
                if (awv >> p) & 1:
                    a = (awaddr >> (AW * p)) & ((1 << AW) - 1)
                    ln = (awlen >> (8 * p)) & 0xFF
                    if (a & 0xFFF) + (ln + 1) * 32 > 4096:
                        self.err(f"port {p} AW crosses 4KB: addr={a:#x} len={ln + 1}")
                    self.aw[p].append([a, ln, (awid >> (IDW * p)) & ((1 << IDW) - 1)])
                    self.port_stats[p].inc("aw")
        if self._wr:
            wv = rv(d.m_axi_wvalid) & self._wr
            if wv:
                wdata, wstrb, wlast = rv(d.m_axi_wdata), rv(d.m_axi_wstrb), rv(d.m_axi_wlast)
                for p in range(N):
                    if (wv >> p) & 1:
                        a = self.aw[p][0]
                        beat = self.wbeat[p]
                        data = (wdata >> (256 * p)) & ((1 << 256) - 1)
                        strb = (wstrb >> (32 * p)) & 0xFFFFFFFF
                        self.mem.write(a[0] + beat * 32, beat_to_bytes(data, 32), strb)
                        self.port_stats[p].inc("w_beats")
                        last = (wlast >> p) & 1
                        if bool(last) != (beat == a[1]):
                            self.err(f"port {p} wlast={last} at beat {beat}/{a[1]}")
                        if beat == a[1]:
                            self.aw[p].popleft()
                            self.wbeat[p] = 0
                            self.b[p].append([a[2], cyc + self.rng.randint(*self.b_lat)])
                        else:
                            self.wbeat[p] = beat + 1
        if self._bv:
            br = rv(d.m_axi_bready) & self._bv
            for p in range(N):
                if (br >> p) & 1:
                    self.b[p].popleft()

    def idle(self):
        return all(not q for q in self.ar) and all(not q for q in self.aw) and all(not q for q in self.b)


# =========================================================================================
# AXI-Lite master (host control)
# =========================================================================================
class AxiLiteMaster(Comp):
    def __init__(self, dut, prefix="s_axi_control", name="axil"):
        super().__init__(name)
        self.p = Port(dut, prefix, ["awaddr", "awvalid", "awready", "wdata", "wstrb", "wvalid", "wready",
                                    "bresp", "bvalid", "bready", "araddr", "arvalid", "arready", "rdata",
                                    "rresp", "rvalid", "rready"])
        self.ops = deque()

    def reset(self):
        self.cur = None
        for s in ("awvalid", "wvalid", "bready", "arvalid", "rready"):
            getattr(self.p, s).value = 0

    async def write(self, a, d, strb=0xF):
        ev = Event()
        self.ops.append({"w": True, "a": a, "d": d, "strb": strb, "ev": ev, "aw": False, "wd": False})
        await ev.wait()

    async def read(self, a):
        ev = Event()
        op = {"w": False, "a": a, "ev": ev, "ar": False}
        self.ops.append(op)
        await ev.wait()
        return op["r"]

    def drive(self):
        p = self.p
        if self.cur is None and self.ops:
            self.cur = self.ops.popleft()
        c = self.cur
        p.awvalid.value = int(bool(c and c["w"] and not c["aw"]))
        p.wvalid.value = int(bool(c and c["w"] and not c["wd"]))
        p.bready.value = int(bool(c and c["w"] and c["aw"] and c["wd"]))
        p.arvalid.value = int(bool(c and not c["w"] and not c["ar"]))
        p.rready.value = int(bool(c and not c["w"] and c["ar"]))
        if c:
            if c["w"]:
                p.awaddr.value, p.wdata.value, p.wstrb.value = c["a"], c["d"], c["strb"]
            else:
                p.araddr.value = c["a"]

    def sample(self):
        c, p = self.cur, self.p
        if not c:
            return
        if c["w"]:
            if c["aw"] and c["wd"] and rv(p.bvalid):
                c["ev"].set()
                self.cur = None
                return
            if not c["aw"] and rv(p.awready):
                c["aw"] = True
            if not c["wd"] and rv(p.wready):
                c["wd"] = True
        else:
            if c["ar"] and rv(p.rvalid):
                c["r"] = rv(p.rdata)
                c["ev"].set()
                self.cur = None
                return
            if not c["ar"] and rv(p.arready):
                c["ar"] = True
