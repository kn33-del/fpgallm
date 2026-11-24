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


class QuantLlama(TinyModel):
    def __init__(self, fm, probes, qshift=8, log=print):
        assert cfg.MODEL == "tinyllama" and cfg.D_MODEL == fm.D and cfg.N_LAYERS == fm.L
        L, D, F, H, KVH, HD = fm.L, fm.D, fm.F, fm.H, fm.KVH, fm.HD
        G = H // KVH
        P = probes
        self.mm_shift = S = cfg.T1_SHIFT
        self.theta = fm.theta
        self.stats = {}
        s_x = P["x"] * 1.05 / 32767.0
        self.s_x = s_x
        self.s_k, self.s_v = [], []
        # ---- global SiLU*up scale: QM chosen so the median layer needs c_l ~ 1
        s_u_want = np.array([P[f"u{l}"] / 127.0 for l in range(L)])
        self.qs = qshift
        self.qm = int(max(1, round((1 << qshift) / float(np.median(s_u_want)))))
        base_su = (1 << qshift) / self.qm
        log(f"[quant] s_x={s_x:.4g} (|x|max={P['x']:.1f})  QUANT_MULT={self.qm} QUANT_SHIFT={self.qs}")

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
            s_h, s_h2 = P[f"h{l}"] / 127.0, P[f"h2_{l}"] / 127.0
            self.g_attn.append(gamma(fm.small(p + "input_layernorm.weight"), s_h))
            self.g_mlp.append(gamma(fm.small(p + "post_attention_layernorm.weight"), s_h2))
            s_q = [P[f"q{l}.{g}"] / 127.0 for g in range(KVH)]
            s_k = [P[f"k{l}.{g}"] / 127.0 for g in range(KVH)]
            s_v = [P[f"v{l}.{g}"] / 127.0 for g in range(KVH)]
            self.s_k.append(s_k)
            self.s_v.append(s_v)
            Wq, sq = q8_per_out(fm.w(p + "self_attn.q_proj.weight"))
            Wk, sk = q8_per_out(fm.w(p + "self_attn.k_proj.weight"))
            Wv, sv = q8_per_out(fm.w(p + "self_attn.v_proj.weight"))
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
            # ---- MLP with per-layer up-scale factor c_l
            c_l = min(base_su / s_u_want[l], 0.95 * 128.0 / max(P[f"up{l}"], 1e-9))
            s_u = base_su / c_l
            Wg, sgw = q8_per_out(fm.w(p + "mlp.gate_proj.weight"))
            Wu, suw = q8_per_out(fm.w(p + "mlp.up_proj.weight"))
            Wd, sdw = q8_per_out(fm.w(p + "mlp.down_proj.weight"))
            self.Wg.append(Wg)
            self.Wu.append(Wu)
            self.sg.append(mult((1 << (S + 8)) * s_h2 * sgw))
            self.su.append(mult((1 << (S + 8)) * s_h2 * suw * c_l))
            self.Wd.append(Wd)
            self.sd.append(mult((1 << S) * s_u * sdw / s_x))
            self._gu.append(TinyModel.gu(self, l))
            self.stats[l] = dict(c_l=round(c_l, 3), s_u=s_u, u_max=P[f"u{l}"], gate_max=P[f"gate{l}"],
                                 up_max=P[f"up{l}"], score_max=P[f"score{l}"])
        s_hf = P["hf"] / 127.0
        self.g_final = gamma(fm.small("model.norm.weight"), s_hf)
        Wl, sl = q8_per_out(fm.w("lm_head.weight"))
        s_lg = P["logits"] * 1.1 / 32767.0
        self.Wlm, self.s_lm = Wl, mult((1 << S) * s_hf * sl / s_lg)
        self.Kc = np.zeros((L, KVH, HD, cfg.MAX_SEQ), np.int64)
        self.Vc = np.zeros((L, KVH, cfg.MAX_SEQ, HD), np.int64)
        for name in ("g_attn", "g_mlp"):
            sat = sum(int(np.sum(np.abs(g) >= 32767)) for g in getattr(self, name))
            if sat:
                log(f"[quant] WARNING {sat} saturated {name} gamma codes")

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
    return P
