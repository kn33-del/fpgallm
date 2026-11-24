#!/usr/bin/env python3
"""Build + run the pyuvm/cocotb regression with Verilator.

  python3 verif/run.py                 # everything
  python3 verif/run.py t1 t3           # selected benches
  python3 verif/run.py t1 -k T1GemmPrefill --seed 7 --waves

Benches: t1 (mm_engine), t2 (attn_unit), t3 (mlp_unit/vec_unit), t4 (mem_subsys+ctrl_regs),
         top (llm_top full pipeline, LLM_TINY dims).
Reports: build/verif/reports/<Test>.json (profiling), build/verif/<bench>/results.xml
"""
import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "verif"
PKG = [ROOT / "rtl/common/llm_pkg.sv", ROOT / "rtl/common/llm_if.sv"]
ALL_RTL = PKG + [ROOT / "rtl/common/llm_ctrl.sv", ROOT / "rtl/t1_compute/t1_compute.sv",
                 ROOT / "rtl/t2_attention/t2_attention.sv", ROOT / "rtl/t3_mlp/t3_mlp.sv",
                 ROOT / "rtl/t4_memory/t4_memory.sv", ROOT / "rtl/llm_top.sv"]

BENCHES = {
    "t1": dict(top="tb_t1", src=PKG + [ROOT / "rtl/t1_compute/t1_compute.sv", V / "hdl/tb_t1.sv"]),
    "t2": dict(top="tb_t2", src=PKG + [ROOT / "rtl/t2_attention/t2_attention.sv", V / "hdl/tb_t2.sv"]),
    "t3": dict(top="tb_t3", src=PKG + [ROOT / "rtl/t3_mlp/t3_mlp.sv", V / "hdl/tb_t3.sv"]),
    "t4": dict(top="tb_t4", src=PKG + [ROOT / "rtl/t4_memory/t4_memory.sv", V / "hdl/tb_t4.sv"]),
    "top": dict(top="llm_top", src=ALL_RTL, tiny=True),
}

WARN_OFF = ["-Wno-fatal", "-Wno-WIDTH", "-Wno-UNOPTFLAT", "-Wno-PINCONNECTEMPTY", "-Wno-EOFNEWLINE",
            "-Wno-SYMRSVDWORD", "-Wno-MULTIDRIVEN", "-Wno-SYNCASYNCNET", "-Wno-TIMESCALEMOD",
            "-Wno-CASEINCOMPLETE", "-Wno-UNSIGNED", "-Wno-CMPCONST", "-Wno-SELRANGE"]


def run_bench(name, args):
    b = BENCHES[name]
    tiny = b.get("tiny", False)
    build_dir = ROOT / "build/verif" / name
    runner = get_runner("verilator")
    defines = {"LLM_TINY": 1} if tiny else {}
    runner.build(
        sources=b["src"], hdl_toplevel=b["top"], includes=[V / "hdl"], defines=defines,
        build_args=WARN_OFF + ["--assert", "-j", str(os.cpu_count() or 4), "-O3",
                               "--x-assign", "fast", "--x-initial", "fast"],
        build_dir=build_dir, waves=args.waves, timescale=("1ns", "1ps"),
        log_file=build_dir / "build.log",
    )
    env = {
        "PYTHONPATH": os.pathsep.join([str(V), str(V / "tests"), os.environ.get("PYTHONPATH", "")]),
        "LLM_TINY": "1" if tiny else "0",
        "LLM_SEED": str(args.seed),
        "LLM_REPORT_DIR": str(ROOT / "build/verif/reports"),
        "COCOTB_LOG_LEVEL": os.environ.get("COCOTB_LOG_LEVEL", "INFO"),
    }
    os.environ.update(env)
    for p in (str(V), str(V / "tests")):   # the runner builds the sim's PYTHONPATH from sys.path
        if p not in sys.path:
            sys.path.insert(0, p)
    xml = build_dir / "results.xml"
    runner.test(
        test_module=f"test_{name}", hdl_toplevel=b["top"], build_dir=build_dir, test_dir=build_dir,
        extra_env=env, testcase=args.k or None, seed=args.seed, waves=args.waves,
        results_xml=str(xml), log_file=build_dir / "sim.log",
    )
    res = []
    for tc in ET.parse(xml).getroot().iter("testcase"):
        failed = tc.find("failure") is not None or tc.find("error") is not None
        res.append((name, tc.get("name"), "FAIL" if failed else ("SKIP" if tc.find("skipped") is not None else "PASS"),
                    float(tc.get("time", 0) or 0)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benches", nargs="*", default=list(BENCHES))
    ap.add_argument("-k", action="append", help="testcase name (repeatable)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--waves", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    results, t0 = [], time.time()
    for b in args.benches:
        try:
            results += run_bench(b, args)
        except Exception as e:  # build failure etc.
            results.append((b, "<build/run>", f"ERROR: {e}".splitlines()[0][:120], 0.0))
    print("\n================ REGRESSION SUMMARY ================")
    for bench, tc, st, t in results:
        print(f"  {bench:4s} {tc:32s} {st:6s} {t:8.1f}s")
    n_bad = sum(1 for r in results if r[2] != "PASS" and r[2] != "SKIP")
    print(f"  {len(results) - n_bad}/{len(results)} passed   ({time.time() - t0:.0f}s)")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
