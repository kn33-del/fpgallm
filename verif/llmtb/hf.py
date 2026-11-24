"""Hugging Face Llama-family checkpoint support with no torch / tokenizers dependency:
  - SafeTensors reader (bf16 -> fp32 on demand, memory-mapped)
  - Llama SentencePiece-BPE tokenizer from tokenizer.json (byte fallback)
  - float32 reference forward (KV cache, GQA, rotate_half RoPE) with activation probes
"""
import json
import struct
from pathlib import Path

import numpy as np


# =========================================================================================
class SafeTensors:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name):
        h = self.header[name]
        a, b = h["data_offsets"]
        raw = self.mm[self.base + a:self.base + b]
        if h["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            return u.view(np.float32).reshape(h["shape"])
        if h["dtype"] == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(h["shape"]).copy()
        if h["dtype"] == "F16":
            return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(h["shape"])
        raise ValueError(h["dtype"])


# =========================================================================================
class Tokenizer:
    """Llama/SentencePiece BPE (Prepend+Replace('▁') normalizer, byte fallback)."""
    SP = "▁"

    def __init__(self, path):
        t = json.loads(Path(path).read_text(encoding="utf-8"))
        m = t["model"]
        self.vocab = m["vocab"]
        self.inv = {v: k for k, v in self.vocab.items()}
        merges = m["merges"]
        self.ranks = {}
        for i, mg in enumerate(merges):
            a, b = mg.split(" ", 1) if isinstance(mg, str) else mg
            self.ranks[(a, b)] = i
        self.special = {a["content"]: a["id"] for a in t.get("added_tokens", []) if a.get("special")}
        self.bos = self.special.get("<s>", 1)
        self.eos = self.special.get("</s>", 2)

    def _bpe(self, text):
        syms = list(text)
        while len(syms) > 1:
            best, bi = None, -1
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            syms[bi:bi + 2] = [syms[bi] + syms[bi + 1]]
        ids = []
        for s in syms:
            if s in self.vocab:
                ids.append(self.vocab[s])
            else:
                ids.extend(self.vocab[f"<0x{b:02X}>"] for b in s.encode("utf-8"))
        return ids

    def encode(self, text, bos=True):
        ids = [self.bos] if bos else []
        # split on special tokens; each plain segment is normalized independently
        parts, i = [], 0
        specials = sorted(self.special, key=len, reverse=True)
        while i < len(text):
            for sp in specials:
                if text.startswith(sp, i):
                    parts.append((True, sp))
                    i += len(sp)
                    break
            else:
                j = i
                while j < len(text) and not any(text.startswith(sp, j) for sp in specials):
                    j += 1
                parts.append((False, text[i:j]))
                i = j
        first = True
        for is_sp, seg in parts:
            if is_sp:
                ids.append(self.special[seg])
            elif seg:
                norm = (self.SP if first else "") + seg.replace(" ", self.SP)
                ids.extend(self._bpe(norm))
            first = False
        return ids

    def decode(self, ids, skip_special=True):
        out, buf = [], bytearray()
        for i in ids:
            tok = self.inv.get(int(i), "")
            if skip_special and tok in self.special:
                continue
            if tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
                buf.append(int(tok[3:5], 16))
                continue
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
            out.append(tok.replace(self.SP, " "))
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def chat(self, user, system=None):
        """TinyLlama-Chat (Zephyr) template with generation prompt."""
        s = ""
        if system:
            s += f"<|system|>\n{system}</s>\n"
        s += f"<|user|>\n{user}</s>\n<|assistant|>\n"
        return self.encode(s)


# =========================================================================================
class FloatLlama:
    """fp32 reference. probes: dict name -> running max |value| (for calibration)."""

    def __init__(self, model_dir):
        d = Path(model_dir)
        self.cfg = json.loads((d / "config.json").read_text())
        self.st = SafeTensors(d / "model.safetensors")
        c = self.cfg
        self.D, self.L = c["hidden_size"], c["num_hidden_layers"]
        self.H, self.KVH = c["num_attention_heads"], c["num_key_value_heads"]
        self.HD = self.D // self.H
        self.F, self.V = c["intermediate_size"], c["vocab_size"]
        self.eps, self.theta = c["rms_norm_eps"], c.get("rope_theta", 10000.0)
        self.inv = self.theta ** (-np.arange(0, self.HD, 2) / self.HD)
        self._small = {}
        self.reset()
        self.probes = None
        self.trace = None
        self._pos = 0

    def reset(self):
        self.K = [[] for _ in range(self.L)]
        self.Vc = [[] for _ in range(self.L)]

    def w(self, name):
        return self.st.get(name)

    def small(self, name):
        if name not in self._small:
            self._small[name] = self.st.get(name)
        return self._small[name]

    def probe(self, name, x, per=None):
        # BOS (pos 0) carries Llama's "massive activations"; it is pre-computed on the host
        # (QuantLlama.seed_bos), so only its K/V ranges matter for calibration.
        if self.probes is None or (self._pos == 0 and name[0] not in "kv"):
            return
        v = float(np.max(np.abs(x)))
        self.probes[name] = max(self.probes.get(name, 0.0), v)
        if per is not None:                       # per-channel running max (SmoothQuant stats)
            a = np.abs(np.asarray(x, np.float64)).reshape(-1)
            key = "ch:" + name
            old = self.probes.get(key)
            self.probes[key] = a if old is None else np.maximum(np.asarray(old), a)
        if name == "x":
            self.probes.setdefault("x_tok", []).append(v)

    def rms(self, x, w):
        return x / np.sqrt(np.mean(x * x) + self.eps) * w

    def rope(self, x, pos):
        P = self.HD // 2
        c, s = np.cos(pos * self.inv), np.sin(pos * self.inv)
        x0, x1 = x[..., :P], x[..., P:]
        return np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)

    def step(self, token, pos):
        self._pos = pos
        H, KVH, HD, G = self.H, self.KVH, self.HD, self.H // self.KVH
        x = self.w("model.embed_tokens.weight")[token].astype(np.float32)
        self.probe("x", x)
        for l in range(self.L):
            p = f"model.layers.{l}."
            h = self.rms(x, self.small(p + "input_layernorm.weight"))
            self.probe(f"h{l}", h, per=True)
            q = (self.w(p + "self_attn.q_proj.weight") @ h).reshape(H, HD)
            k = (self.w(p + "self_attn.k_proj.weight") @ h).reshape(KVH, HD)
            v = (self.w(p + "self_attn.v_proj.weight") @ h).reshape(KVH, HD)
            q, k = self.rope(q, pos), self.rope(k, pos)
            for g in range(KVH):
                self.probe(f"q{l}.{g}", q[g * G:(g + 1) * G])
                self.probe(f"k{l}.{g}", k[g])
                self.probe(f"v{l}.{g}", v[g])
            self.K[l].append(k)
            self.Vc[l].append(v)
            Kl, Vl = np.stack(self.K[l], 1), np.stack(self.Vc[l], 1)     # [KVH, T, HD]
            o = np.zeros((H, HD), np.float32)
            for hh in range(H):
                g = hh // G
                sc = Kl[g] @ q[hh] / np.sqrt(HD)
                self.probe(f"score{l}", sc)
                e = np.exp(sc - sc.max())
                o[hh] = (e / e.sum()) @ Vl[g]
            self.probe(f"head{l}", o, per=True)
            a = self.w(p + "self_attn.o_proj.weight") @ o.reshape(-1)
            x = x + a
            self.probe("x", x)
            h = self.rms(x, self.small(p + "post_attention_layernorm.weight"))
            self.probe(f"h2_{l}", h, per=True)
            gt = self.w(p + "mlp.gate_proj.weight") @ h
            up = self.w(p + "mlp.up_proj.weight") @ h
            self.probe(f"gate{l}", gt)
            self.probe(f"up{l}", up, per=True)
            u = gt / (1 + np.exp(-gt)) * up
            self.probe(f"u{l}", u, per=True)
            x = x + self.w(p + "mlp.down_proj.weight") @ u
            self.probe("x", x)
            if self.trace is not None:
                self.trace[f"x_l{l}"] = x.copy()
        h = self.rms(x, self.small("model.norm.weight"))
        self.probe("hf", h)
        logits = self.w("lm_head.weight") @ h
        self.probe("logits", logits)
        return logits

    def generate(self, ids, n_new, stop=None, on_token=None):
        self.reset()
        out = []
        for pos, t in enumerate(ids):
            logits = self.step(t, pos)
        for i in range(n_new):
            t = int(np.argmax(logits))
            out.append(t)
            if on_token:
                on_token(t)
            if stop is not None and t == stop:
                break
            logits = self.step(t, len(ids) + i)
        return out
