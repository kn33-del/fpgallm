#!/usr/bin/env python3
"""CPU demo: TinyLlama-1.1B-Chat through the bit-exact integer model of the FPGA datapath.

  python scripts/cpu_demo.py --model ../TinyLlama-1.1B-Chat-v1.0 "What is an FPGA?"

The integer path (verif/llmtb/model.py step + golden.py) is exactly what the RTL computes,
so the text printed here is what the U55C should print token-for-token. --float also runs
the fp32 reference for comparison. Calibration scales are cached in build/tinyllama_calib.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LLM_MODEL", "tinyllama")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verif"))

from llmtb.hf import FloatLlama, Tokenizer          # noqa: E402
from llmtb.quantize import CALIB_PROMPTS, QuantLlama, calibrate    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="What is an FPGA? Answer in one sentence.")
    ap.add_argument("--model", default=str(ROOT.parent / "TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--float", action="store_true")
    ap.add_argument("--recalibrate", action="store_true")
    a = ap.parse_args()
    tok = Tokenizer(Path(a.model) / "tokenizer.json")
    fm = FloatLlama(a.model)
    cal = ROOT / "build/tinyllama_calib.json"
    if cal.exists() and not a.recalibrate:
        P = json.loads(cal.read_text())
    else:
        t0 = time.time()
        P = calibrate(fm, tok, CALIB_PROMPTS)
        cal.parent.mkdir(parents=True, exist_ok=True)
        cal.write_text(json.dumps(P, indent=1))
        print(f"[calib] {time.time() - t0:.0f}s -> {cal}")
    t0 = time.time()
    qm = QuantLlama(fm, P)
    print(f"[quant] built in {time.time() - t0:.0f}s; RTL params: QUANT_MULT={qm.qm} QUANT_SHIFT={qm.qs}")
    qm.seed_bos(fm, tok.bos)
    ids = tok.chat(a.prompt)
    print(f"prompt ({len(ids)} tokens): {a.prompt}")
    if a.float:
        t0 = time.time()
        out = fm.generate(ids, a.n, stop=tok.eos)
        print(f"\n[fp32]      {tok.decode(out)}   ({time.time() - t0:.0f}s)")
    t0 = time.time()
    print("\n[FPGA math] ", end="", flush=True)
    shown = []

    def show(t):
        shown.append(t)
        sys.stdout.write("\r[FPGA math] " + tok.decode(shown))
        sys.stdout.flush()
    out = qm.generate(ids, a.n, stop=tok.eos, on_token=show)
    print(f"\n({len(out)} tokens, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
