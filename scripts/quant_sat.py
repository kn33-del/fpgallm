#!/usr/bin/env python3
"""Per-layer saturation report for the integer TinyLlama datapath (one token, after a prefix)."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("LLM_MODEL", "tinyllama")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verif"))
import numpy as np
from llmtb import cfg, golden
from llmtb.hf import FloatLlama, Tokenizer
from llmtb.quantize import QuantLlama
from llmtb.model import rope_table

M = str(ROOT.parent / "TinyLlama-1.1B-Chat-v1.0")
tok = Tokenizer(M + "/tokenizer.json"); fm = FloatLlama(M)
P = json.loads((ROOT / "build/tinyllama_calib.json").read_text())
logs = []
qm = QuantLlama(fm, P, log=logs.append)
print("\n".join(logs))
for l in range(cfg.N_LAYERS):
    print(f"gamma l{l}: attn sat {int(np.sum(np.abs(qm.g_attn[l])>=32767))} max {int(np.abs(qm.g_attn[l]).max())} "
          f"min|.| {int(np.abs(qm.g_attn[l]).min())}  mlp sat {int(np.sum(np.abs(qm.g_mlp[l])>=32767))} "
          f"max {int(np.abs(qm.g_mlp[l]).max())}  c_l={qm.stats[l]['c_l']}") if l in (0,1,6,7,8) else None
ids = tok.chat("What is an FPGA? Answer in one sentence.")
qm.seed_bos(fm, tok.bos); qm.reset()
S = qm.mm_shift
for pos, t in list(enumerate(ids[:10]))[1:]:
    r = qm.step(t, pos)
# instrumented re-run of the last token's layers 0..8 using the cached state (cache already has pos)
pos, t = 9, ids[9]
x = qm.emb[t].astype(np.int64)
print("emb codes |max|", np.abs(x).max(), "mean|.|", np.abs(x).mean())
H, KVH, HD = cfg.N_HEADS, cfg.N_KV_HEADS, cfg.HEAD_DIM
c, s = rope_table(pos, qm.theta)
def sat(a, lim): a = np.asarray(a); return int(np.sum(np.abs(a) >= lim))
for l in range(9):
    h = golden.rmsnorm(x, qm.g_attn[l], qm.qm, qm.qs)
    qkv = golden.gemm(h[None], qm.Wqkv[l], qm.s_qkv[l], cfg.OUT_VEC, S)[0]
    q8 = np.concatenate([golden.rope_head(qkv[i*HD:(i+1)*HD], c, s, pos, i < H+KVH) for i in range(H+2*KVH)])
    heads = np.zeros(cfg.D_MODEL, np.int64); scs = []
    for hh in range(H):
        g = hh // (H // KVH)
        sc = golden.gemm(q8[hh*HD:(hh+1)*HD][None], qm.Kc[l, g][:, :pos+1], qm.ks[l][g][:pos+1], cfg.OUT_VEC, S)[0]
        scs.append(sc)
        p = golden.softmax_i8(sc)
        heads[hh*HD:(hh+1)*HD] = golden.gemm(p[None], qm.Vc[l, g][:pos+1], qm.vs[l][g], cfg.OUT_I8, S)[0]
    att = golden.gemm(heads[None], qm.Wo[l], qm.s_wo[l], cfg.OUT_VEC, S)[0]
    x1 = golden.resadd(x, att)
    h2 = golden.rmsnorm(x1, qm.g_mlp[l], qm.qm, qm.qs)
    gu = golden.gemm(h2[None], qm.gu(l)[0], qm.gu(l)[1], cfg.OUT_VEC, S)[0]
    u = golden.silu_mul_interleaved(gu, cfg.D_FF, qm.qm, qm.qs)
    m = golden.gemm(u[None], qm.Wd[l], qm.sd[l], cfg.OUT_VEC, S)[0]
    x = golden.resadd(x1, m)
    print(f"l{l}: h sat {sat(h,127)} |h|max {np.abs(h).max()} | qkv sat {sat(qkv,32767)} | q8 sat {sat(q8,127)} "
          f"| score |max| {np.abs(np.array(scs)).max()} | heads sat {sat(heads,127)} max {np.abs(heads).max()} "
          f"| att |max| {np.abs(att).max()} | h2 sat {sat(h2,127)} | gu sat {sat(gu,32767)} |gu|max {np.abs(gu).max()} "
          f"| u sat {sat(u,127)} |u|max {np.abs(u).max()} nz {int(np.sum(u!=0))} | m |max| {np.abs(m).max()} | x |max| {np.abs(x).max()}")
