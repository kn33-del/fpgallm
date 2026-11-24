#!/usr/bin/env python3
"""Float vs bit-exact integer TinyLlama: per-layer residual error and logits agreement."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("LLM_MODEL", "tinyllama")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verif"))
import numpy as np
from llmtb.hf import FloatLlama, Tokenizer
from llmtb.quantize import QuantLlama

M = str(ROOT.parent / "TinyLlama-1.1B-Chat-v1.0")
tok = Tokenizer(M + "/tokenizer.json")
fm = FloatLlama(M)
P = json.loads((ROOT / "build/tinyllama_calib.json").read_text())
qm = QuantLlama(fm, P, log=lambda *a: None)
s_x = qm.s_x
ids = tok.chat(sys.argv[1] if len(sys.argv) > 1 else "What is an FPGA? Answer in one sentence.")
n = int(os.environ.get("NTOK", "8"))
qm.seed_bos(fm, tok.bos)
fm.reset(); qm.reset()
fm.step(ids[0], 0)
for pos, t in list(enumerate(ids[:n]))[1:]:
    fm.trace = {}
    lf = fm.step(t, pos)
    r = qm.step(t, pos)
print(f"after {n} prompt tokens (pos {n-1}):")
for l in range(fm.L):
    xf, xq = fm.trace[f"x_l{l}"], r[f"x_l{l}"] * s_x
    rel = np.linalg.norm(xq - xf) / np.linalg.norm(xf)
    cos = float(xq @ xf / (np.linalg.norm(xq) * np.linalg.norm(xf)))
    print(f"  layer {l:2d}: rel_err {rel:6.3f}  cos {cos:6.4f}  |x|max f {np.abs(xf).max():7.2f} q {np.abs(xq).max():7.2f}")
lq = r["logits"].astype(float)
print("top5 float:", np.argsort(-lf)[:5], " top5 int:", np.argsort(-lq)[:5],
      " cos(logits)", float(lq @ lf / np.linalg.norm(lq) / np.linalg.norm(lf)))
