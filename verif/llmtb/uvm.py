"""pyuvm layer: sequence items, sequences, a generic command agent, scoreboard, base env/test.

Structure (per DUT):
  BaseTest (uvm_test)
    └─ env (subclass of BaseEnv: uvm_env)
         ├─ ClockDomain + reactive BFMs (memory servers, engine model, HBM)   [llmtb.bfm]
         ├─ CmdAgent(s)  : uvm_sequencer -> CmdDriver(uvm_driver) -> BFM ; CmdMonitor -> analysis port
         ├─ Scoreboard   : uvm_subscriber; compares each completed command against a golden callback
         └─ Profile      : counters dumped in report_phase
"""
import os

import cocotb
from pyuvm import (ConfigDB, uvm_analysis_port, uvm_component, uvm_driver, uvm_env, uvm_monitor,
                   uvm_root, uvm_sequence, uvm_sequence_item, uvm_sequencer, uvm_subscriber, uvm_test)

from .bfm import ClockDomain
from .perf import Profile


class CmdItem(uvm_sequence_item):
    """One command on a valid/ready port. `cmd` is the packed int, `fields` the decoded dict."""

    def __init__(self, name="cmd", cmd=0, fields=None, check=None, label=""):
        super().__init__(name)
        self.cmd = cmd
        self.fields = fields or {}
        self.check = check          # callable(item) -> list[str] of mismatches, run by scoreboard
        self.label = label
        self.acc_cycle = self.done_cycle = None
        self.result = None

    def __str__(self):
        return f"{self.get_name()}[{self.label}] {self.fields}"


class ListSeq(uvm_sequence):
    """Plays a list of pre-built CmdItems in order."""

    def __init__(self, name="list_seq", items=()):
        super().__init__(name)
        self.items = list(items)

    async def body(self):
        for it in self.items:
            await self.start_item(it)
            await self.finish_item(it)


class CmdDriver(uvm_driver):
    def build_phase(self):
        self.bfm = ConfigDB().get(self, "", "BFM")

    async def run_phase(self):
        while True:
            it = await self.seq_item_port.get_next_item()
            t = {}
            ev = self.bfm.send(it.cmd, meta=it)
            await ev.wait()
            t = it  # completion info is filled by the BFM's log entry
            entry = self.bfm.log[-1] if self.bfm.log else {}
            for e in reversed(self.bfm.log):
                if e.get("meta") is it:
                    entry = e
                    break
            it.acc_cycle = entry.get("acc_cycle")
            it.done_cycle = entry.get("done_cycle")
            it.result = entry.get("result")
            self.seq_item_port.item_done()
            self.ap.write(t)

    def connect_phase(self):
        pass


class CmdAgent(uvm_component):
    def build_phase(self):
        self.seqr = uvm_sequencer("seqr", self)
        self.driver = CmdDriver("driver", self)
        self.driver.ap = uvm_analysis_port("ap", self.driver)
        self.ap = self.driver.ap

    def connect_phase(self):
        self.driver.seq_item_port.connect(self.seqr.seq_item_export)


class Scoreboard(uvm_subscriber):
    def build_phase(self):
        self.n_checked = 0
        self.errors = []

    def write(self, item):
        if item.check is None:
            return
        errs = item.check(item) or []
        self.n_checked += 1
        if errs:
            for e in errs[:10]:
                self.logger.error(f"{item.label}: {e}")
            self.errors.extend(f"{item.label}: {e}" for e in errs)
        else:
            self.logger.info(f"PASS {item.label}")


class BaseEnv(uvm_env):
    """Subclasses implement make_bfms(dut, domain) and may add agents in build_phase."""
    clk_name, rst_name = "clk", "rst_n"

    def build_phase(self):
        self.dut = cocotb.top
        self.seed = int(os.environ.get("LLM_SEED", "1"))
        self.domain = ClockDomain(self.dut, self.clk_name)
        self.sb = Scoreboard("sb", self)
        self.agents = {}
        self.make_bfms()

    def add_agent(self, name, bfm):
        self.domain.add(bfm)
        ConfigDB().set(self, f"{name}.driver", "BFM", bfm)
        ag = CmdAgent(name, self)
        self.agents[name] = ag
        return ag

    def connect_phase(self):
        for ag in self.agents.values():
            ag.ap.connect(self.sb.analysis_export)

    def make_bfms(self):
        raise NotImplementedError

    def all_comps(self):
        return self.domain.comps


class BaseTest(uvm_test):
    env_cls = BaseEnv
    timeout_cycles = 3_000_000

    def build_phase(self):
        self.env = self.env_cls("env", self)
        self.profile = Profile(type(self).__name__)

    async def run_phase(self):
        self.raise_objection()
        await self.env.domain.start(self.env.rst_name)
        try:
            await self.body()
        finally:
            await self.env.domain.cycles(20)
            self.drop_objection()

    async def body(self):
        raise NotImplementedError

    async def play(self, agent, items):
        seq = ListSeq("seq", items)
        await seq.start(self.env.agents[agent].seqr)

    def extract_phase(self):          # runs before check_phase, so a failing test still profiles
        cyc = self.env.domain.cycle
        self.profile.add("sim", cycles=cyc, ns=cyc * self.env.domain.period_ns)
        for c in self.env.all_comps():
            if c.stats:
                self.profile.add_port(c, cyc)
        self.extra_profile()
        self.profile.dump(self.logger.info)

    def extra_profile(self):
        pass

    def check_phase(self):
        errs = list(self.env.sb.errors)
        for c in self.env.all_comps():
            errs.extend(c.errors)
        errs.extend(self.extra_checks())
        for e in errs[:40]:
            self.logger.error(e)
        assert not errs, f"{len(errs)} verification error(s); first: {errs[0]}"
        self.logger.info(f"{type(self).__name__}: scoreboard checked {self.env.sb.n_checked} item(s), 0 errors")

    def extra_checks(self):
        return []
