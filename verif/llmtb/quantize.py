"""Post-training quantization of a HF Llama checkpoint into the RTL's exact formats.

Scales are "real value per integer code". Calibration = max |activation| over a few chat
prompts from the float reference (hf.FloatLlama). Formats (see docs/T2.md, t3_mlp.sv):
  residual x        int16, scale s_x (global)          embeddings / Wo / Wd outputs
  normed h          int8,  per-site scale              RMSNorm gamma codes absorb it
  QKV proj          int16 "Q7 of int8 code"            M = 2^23 * s_h * s_w / s_grp
  q,k,v after rope  int8, Q per GQA group, K/V per kv head
  scores            int16 Q8                           M = 2^21 * s_q * s_k   (1/sqrt(64) folded)
  probs             0..127 = p*127
  heads             int8, per-layer s_head             M = 2^16 * s_v / (127 * s_head)
  gate/up           int16 Q8 (value*256)               M = 2^24 * s_h2 * s_w  (* c_l for up)
  u = silu(g)*up    int8, s_u(l) = 2^QS / (QM * c_l)   QM/QS are RTL params (one global pair);
                                                       per-layer c_l is folded into the up scales
  Wd / Wo out       int16 residual codes               M = 2^16 * s_in * s_w / s_x
  RMSNorm gamma     int16 = w * 256 * 2^QS / (QM * s_h)
"""
import numpy as np

from . import cfg
from .model import TinyModel

M_MAX = (1 << 31) - 1
CALIB_PROMPTS = ["What is an FPGA? Answer in one sentence.",
                 "Write a short poem about the ocean.",
                 "Explain how a transformer language model generates text."]


def mult(x):
    m = np.rint(np.asarray(x, dtype=np.float64))
    return np.clip(m, 1, M_MAX).astype(np.int64)


def q8_per_out(W):
    """HF weight [out, in] -> (int8 K x N, per-output-channel scale)."""
    s = np.maximum(np.abs(W).max(axis=1), 1e-12) / 127.0
    Wq = np.clip(np.rint(W / s[:, None]), -127, 127).astype(np.int8)
    return Wq.T.copy(), s.astype(np.float64)


def i16(x):
    return np.clip(np.rint(x), -32768, 32767).astype(np.int64)


def smooth_factors(amax, W_cols_max, alpha=0.5):
    """SmoothQuant: per-input-channel a_j; activation_j /= a_j, weight column j *= a_j."""
    if alpha is None:
        return np.ones_like(np.asarray(amax, np.float64))
    a = np.power(np.maximum(amax, 1e-6), alpha) / np.power(np.maximum(W_cols_max, 1e-6), 1 - alpha)
    return np.clip(a, 1e-3, 1e3)


class QuantLlama(TinyModel):
    # Defaults chosen by sweep (scripts/quant_debug.py): logits cos ~0.98 vs fp32, top-4 agree.
    def __init__(self, fm, probes, qshift=8, alpha=0.65, xclip_pct=None, gamma_target=30000, qm=8000,
                 smooth_u=True, log=print):
        assert cfg.MODEL == "tinyllama" and cfg.D_MODEL == fm.D and cfg.N_LAYERS == fm.L
        L, D, F, H, KVH, HD = fm.L, fm.D, fm.F, fm.H, fm.KVH, fm.HD
        G = H // KVH
        P = probes
        ch = lambda k: np.asarray(P["ch:" + k], np.float64)
        self.mm_shift = S = cfg.T1_SHIFT
        self.theta = fm.theta
        self.stats = {}
        # ---- residual: percentile clip (rare massive activations saturate instead of costing bits)
        xclip = (float(np.percentile(P["x_tok"], xclip_pct)) * 1.25 if xclip_pct is not None
                 else P["x"] * 1.05)
        s_x = xclip / 32767.0
        self.s_x = s_x
        self.s_k, self.s_v = [], []
        fold = {}                                       # SmoothQuant-folded float weights per layer
        for l in range(L):
            p = f"model.layers.{l}."
            Wq, Wk, Wv = (fm.w(p + f"self_attn.{n}_proj.weight") for n in "qkv")
            Wg, Wu, Wd = (fm.w(p + f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
            a1 = smooth_factors(ch(f"h{l}"), np.maximum.reduce([np.abs(W).max(0) for W in (Wq, Wk, Wv)]), alpha)
            a2 = smooth_factors(ch(f"h2_{l}"), np.maximum(np.abs(Wg).max(0), np.abs(Wu).max(0)), alpha)
            a3 = smooth_factors(ch(f"u{l}"), np.abs(Wd).max(0), alpha if smooth_u else None)
            fold[l] = dict(a1=a1, a2=a2, a3=a3,
                           g1=fm.small(p + "input_layernorm.weight") / a1,
                           g2=fm.small(p + "post_attention_layernorm.weight") / a2,
                           s_h=float(np.max(ch(f"h{l}") / a1)) / 127.0,
                           s_h2=float(np.max(ch(f"h2_{l}") / a2)) / 127.0,
                           s_u=float(np.max(ch(f"u{l}") / a3)) / 127.0,
                           up_ok=float(np.min(0.95 * 128.0 * a3 / np.maximum(ch(f"up{l}"), 1e-9))))
        s_hf = P["hf"] / 127.0
        gf = fm.small("model.norm.weight")
        # ---- one global QUANT_MULT: the largest folded gamma lands at ~gamma_target codes
        need = max([float(np.max(np.abs(f["g1"]))) / f["s_h"] for f in fold.values()] +
                   [float(np.max(np.abs(f["g2"]))) / f["s_h2"] for f in fold.values()] +
                   [float(np.max(np.abs(gf))) / s_hf])
        self.qs = qshift
        self.qm = int(qm) if qm else int(max(1, np.floor(need * 256.0 * (1 << qshift) / gamma_target)))
        base_su = (1 << qshift) / self.qm
        log(f"[quant] s_x={s_x:.4g} (clip {xclip:.1f}, max seen {P['x']:.1f})  "
            f"QUANT_MULT={self.qm} QUANT_SHIFT={self.qs}  alpha={alpha}")

        def gamma(w, s_h):
            return i16(w * 256.0 * (1 << self.qs) / (self.qm * s_h))

        self.emb = i16(fm.w("model.embed_tokens.weight") / s_x)
        self.g_attn, self.g_mlp = [], []
        self.Wqkv, self.s_qkv, self.Wo, self.s_wo = [], [], [], []
        self.ks, self.vs = [], []
        self.Wg, self.Wu, self.sg, self.su, self.Wd, self.sd = [], [], [], [], [], []
        self._gu = []
        for l in range(L):
            p = f"model.layers.{l}."
            f = fold[l]
            s_h, s_h2 = f["s_h"], f["s_h2"]
            self.g_attn.append(gamma(f["g1"], s_h))
            self.g_mlp.append(gamma(f["g2"], s_h2))
            s_q = [P[f"q{l}.{g}"] / 127.0 for g in range(KVH)]
            s_k = [P[f"k{l}.{g}"] / 127.0 for g in range(KVH)]
            s_v = [P[f"v{l}.{g}"] / 127.0 for g in range(KVH)]
            self.s_k.append(s_k)
            self.s_v.append(s_v)
            Wq, sq = q8_per_out(fm.w(p + "self_attn.q_proj.weight") * f["a1"])
            Wk, sk = q8_per_out(fm.w(p + "self_attn.k_proj.weight") * f["a1"])
            Wv, sv = q8_per_out(fm.w(p + "self_attn.v_proj.weight") * f["a1"])
            grp = np.concatenate([np.repeat([s_q[hh // G] for hh in range(H)], HD),
                                  np.repeat(s_k, HD), np.repeat(s_v, HD)])
            self.Wqkv.append(np.concatenate([Wq, Wk, Wv], axis=1))
            self.s_qkv.append(mult((1 << (S + 7)) * s_h * np.concatenate([sq, sk, sv]) / grp))
            s_head = P[f"head{l}"] / 127.0
            self.ks.append([np.full(cfg.MAX_SEQ, mult((1 << (S + 5)) * s_q[g] * s_k[g])) for g in range(KVH)])
            self.vs.append([np.full(HD, mult((1 << S) * s_v[g] / (127.0 * s_head))) for g in range(KVH)])
            Wo, so = q8_per_out(fm.w(p + "self_attn.o_proj.weight"))
            self.Wo.append(Wo)
            self.s_wo.append(mult((1 << S) * s_head * so / s_x))
            # ---- MLP: u_j /= a3_j (per-channel up multiplier), Wd column j *= a3_j;
            #      per-layer factor c_l maps the global QUANT_MULT onto this layer's u scale.
            c_l = min(base_su / f["s_u"], f["up_ok"])
            s_u = base_su / c_l
            Wg, sgw = q8_per_out(fm.w(p + "mlp.gate_proj.weight") * f["a2"])
            Wu, suw = q8_per_out(fm.w(p + "mlp.up_proj.weight") * f["a2"])
            Wd, sdw = q8_per_out(fm.w(p + "mlp.down_proj.weight") * f["a3"])
            self.Wg.append(Wg)
            self.Wu.append(Wu)
            self.sg.append(mult((1 << (S + 8)) * s_h2 * sgw))
            self.su.append(mult((1 << (S + 8)) * s_h2 * suw * c_l / f["a3"]))
            self.Wd.append(Wd)
            self.sd.append(mult((1 << S) * s_u * sdw / s_x))
            self._gu.append(TinyModel.gu(self, l))
            self.stats[l] = dict(c_l=round(c_l, 3), s_u=s_u, s_u_want=f["s_u"], up_limited=c_l < base_su / f["s_u"],
                                 gmin1=int(np.abs(self.g_attn[-1]).min()), gmax1=int(np.abs(self.g_attn[-1]).max()))
        self.g_final = gamma(gf, s_hf)
        Wl, sl = q8_per_out(fm.w("lm_head.weight"))
        s_lg = P["logits"] * 1.1 / 32767.0
        self.Wlm, self.s_lm = Wl, mult((1 << S) * s_hf * sl / s_lg)
        self.Kc = np.zeros((L, KVH, HD, cfg.MAX_SEQ), np.int64)
        self.Vc = np.zeros((L, KVH, cfg.MAX_SEQ, HD), np.int64)
        for name in ("g_attn", "g_mlp"):
            sat = sum(int(np.sum(np.abs(g) >= 32767)) for g in getattr(self, name))
            if sat:
                log(f"[quant] WARNING {sat} saturated {name} gamma codes")
        lim = [l for l in range(L) if self.stats[l]["up_limited"]]
        if lim:
            log(f"[quant] layers whose u scale is limited by up-proj int16 range: {lim}")

    def gu(self, l):
        return self._gu[l]

    def seed_bos(self, fm, bos):
        """Float-compute the BOS token (pos 0) on the host; store its K/V (quantized with the
        hardware's K/V scales) as pos 0 of the KV cache. The FPGA starts at pos 1."""
        fm.reset()
        fm.step(bos, 0)
        self.bos_k = np.zeros((cfg.N_LAYERS, cfg.N_KV_HEADS, cfg.HEAD_DIM), np.int64)
        self.bos_v = np.zeros_like(self.bos_k)
        for l in range(cfg.N_LAYERS):
            for g in range(cfg.N_KV_HEADS):
                self.bos_k[l, g] = np.clip(np.rint(fm.K[l][0][g] / self.s_k[l][g]), -128, 127)
                self.bos_v[l, g] = np.clip(np.rint(fm.Vc[l][0][g] / self.s_v[l][g]), -128, 127)
        fm.reset()

    def reset(self):
        self.Kc[:] = 0
        self.Vc[:] = 0
        if hasattr(self, "bos_k"):
            self.Kc[:, :, :, 0] = self.bos_k
            self.Vc[:, :, 0, :] = self.bos_v

    def write_bos_kv(self, hbm):
        """Place the seeded pos-0 K/V into the HBM KV cache (llm_pkg kv_tag layout)."""
        for l in range(cfg.N_LAYERS):
            for g in range(cfg.N_KV_HEADS):
                kb, vb = cfg.kv_pc_off(l, g, 0), cfg.kv_pc_off(l, g, 1)
                for d in range(cfg.HEAD_DIM):
                    hbm.write(((d % 32) << 29) | (kb + (d // 32) * 32), np.int8(self.bos_k[l, g, d]).tobytes())
                    hbm.write((0 << 29) | (vb + (d // 32) * (cfg.MAX_SEQ // 32) * 32 + d % 32),
                              np.int8(self.bos_v[l, g, d]).tobytes())

    def generate(self, ids, n_new, stop=None, on_token=None):
        """ids[0] must be BOS (seeded by seed_bos); the integer datapath runs from pos 1."""
        self.reset()
        for pos, t in list(enumerate(ids))[1:]:
            r = self.step(t, pos)
        out = []
        for i in range(n_new):
            t = r["token"]
            out.append(t)
            if on_token:
                on_token(t)
            if stop is not None and t == stop:
                break
            r = self.step(t, len(ids) + i)
        return out


def calibrate(fm, tok, prompts, n_new=24, log=print):
    fm.probes = {}
    for pr in prompts:
        ids = tok.chat(pr)
        out = fm.generate(ids, n_new, stop=tok.eos)
        log(f"[calib] {pr!r} -> {tok.decode(out)!r}")
    P, fm.probes = fm.probes, None
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in P.items()}
