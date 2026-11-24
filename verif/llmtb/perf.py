"""Profiling: per-port / per-command counters -> table on stdout + JSON under build/verif/reports."""
import json
import os
from pathlib import Path

from . import cfg

REPORT_DIR = Path(os.environ.get("LLM_REPORT_DIR", "build/verif/reports"))


class Profile:
    def __init__(self, test_name):
        self.test = test_name
        self.sections = {}

    def add(self, section, **kv):
        self.sections.setdefault(section, {}).update(kv)

    def add_port(self, comp, cycles):
        s = dict(comp.stats)
        busy = s.get("busy_cycles", 0)
        beats = s.get("beats", 0)
        if cycles:
            s["occupancy"] = round(busy / cycles, 4)
        if busy:
            s["beats_per_busy_cycle"] = round(beats / busy, 4)
        self.sections[f"port:{comp.name}"] = s

    @staticmethod
    def gemm_metrics(c, acc_cycle, done_cycle):
        kt, nt = cfg.ceil_div(c["k"], cfg.ARR_R), cfg.ceil_div(c["n"], cfg.ARR_C)
        slices = kt * nt * c["m"]               # one K-slice issue per array cycle
        cyc = max(done_cycle - acc_cycle, 1)
        wbytes = kt * nt * cfg.TILE_BYTES
        return {
            "m": c["m"], "k": c["k"], "n": c["n"], "cycles": cyc,
            "array_slices": slices,
            "array_util": round(slices / cyc, 4),
            "macs_per_cycle": round(slices * cfg.ARR_R * cfg.ARR_C / cyc, 1),
            "weight_GBps_at_300MHz": round(wbytes / cyc * cfg.KERNEL_MHZ * 1e6 / 1e9, 2),
            "pct_of_port_ceiling": round(100 * wbytes / cyc * cfg.KERNEL_MHZ * 1e6 / 1e9 / cfg.PORT_CEIL_GBPS, 1),
        }

    def dump(self, log=print):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{self.test}.json"
        path.write_text(json.dumps(self.sections, indent=2, default=str))
        log(f"==== PROFILE {self.test} ====")
        for sec, kv in self.sections.items():
            items = ", ".join(f"{k}={v}" for k, v in kv.items() if not isinstance(v, (list, dict)))
            log(f"  {sec:28s} {items}")
        log(f"  -> {path}")
        return path
