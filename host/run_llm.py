#!/usr/bin/env python3
"""U55C host: chat with TinyLlama running on the llm_kernel FPGA design (XRT Python bindings).

  source /opt/xilinx/xrt/setup.sh
  python3 host/run_llm.py --xclbin llm_kernel.xclbin --model ~/TinyLlama-1.1B-Chat-v1.0 "What is an FPGA?"

Flow: quantize checkpoint (same code as the CPU demo / simulations) -> pack HBM image per the
llm_pkg contracts -> one 32-bank buffer per m{p}_axi -> per token: set token_pos/token_in,
start, wait, read result_token (0x20) -> detokenize and print. --check also runs the
bit-exact CPU model and stops at the first token that differs.
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

import numpy as np                                   # noqa: E402

from llmtb import cfg                                # noqa: E402
from llmtb.hf import FloatLlama, Tokenizer           # noqa: E402
from llmtb.quantize import CALIB_PROMPTS, QuantLlama, calibrate     # noqa: E402

PC_BYTES = 1 << 29
PAGE = 1 << 20
N_SCALAR_ARGS = 5            # instr_base, n_instr, token_pos, result_token, token_in


class BankedImage:
    """Collects packer writes at kernel addresses {pc, off} as sparse 1 MB pages per bank."""

    def __init__(self):
        self.pages = [dict() for _ in range(cfg.N_HBM)]
        self.hi = [0] * cfg.N_HBM

    def write(self, addr, data, strb=None):
        data = memoryview(bytes(data))
        pc, off = addr >> 29, addr & (PC_BYTES - 1)
        assert pc < cfg.N_HBM and off + len(data) <= PC_BYTES, f"write crosses a bank: {addr:#x}+{len(data)}"
        self.hi[pc] = max(self.hi[pc], off + len(data))
        i = 0
        while i < len(data):
            pg, o = divmod(off + i, PAGE)
            k = min(len(data) - i, PAGE - o)
            buf = self.pages[pc].setdefault(pg, bytearray(PAGE))
            buf[o:o + k] = data[i:i + k]
            i += k


def kv_end():
    return cfg.KV_PC_BASE + cfg.N_LAYERS * cfg.N_KV_HEADS * 2 * cfg.KV_BLK_BYTES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="What is an FPGA? Answer in one sentence.")
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--model", default=str(ROOT.parent / "TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("--device", default="0", help="device index or BDF")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--check", action="store_true", help="compare every token with the CPU golden")
    a = ap.parse_args()

    import pyxrt
    tok = Tokenizer(Path(a.model) / "tokenizer.json")
    fm = FloatLlama(a.model)
    cal = ROOT / "build/tinyllama_calib.json"
    if cal.exists():
        P = json.loads(cal.read_text())
    else:
        P = calibrate(fm, tok, CALIB_PROMPTS)
        cal.parent.mkdir(parents=True, exist_ok=True)
        cal.write_text(json.dumps(P, indent=1))
    t0 = time.time()
    qm = QuantLlama(fm, P)
    print(f"[host] quantized in {time.time() - t0:.0f}s (bitstream must use T3_QUANT_MULT={qm.qm}, "
          f"T3_QUANT_SHIFT={qm.qs})")

    qm.seed_bos(fm, tok.bos)            # BOS (pos 0) runs in float on the host; FPGA starts at pos 1
    img = BankedImage()
    A = qm.pack(img, range(cfg.MAX_SEQ))
    qm.write_bos_kv(img)
    print(f"[host] packed {sum(len(p) for p in img.pages)} MB of HBM pages, {qm.n_instr} instructions")

    dev = pyxrt.device(int(a.device) if a.device.isdigit() else a.device)
    uuid = dev.load_xclbin(pyxrt.xclbin(a.xclbin))
    krn = pyxrt.kernel(dev, uuid, "llm_kernel", pyxrt.kernel.cu_access_mode.exclusive)
    bos = []
    for p in range(cfg.N_HBM):
        size = max(img.hi[p], kv_end()) + PAGE
        bo = pyxrt.bo(dev, size, pyxrt.bo.normal, krn.group_id(N_SCALAR_ARGS + p))
        for pg, buf in img.pages[p].items():
            bo.write(np.frombuffer(buf, np.uint8), pg * PAGE)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, 0)
        bos.append(bo)
    print("[host] HBM loaded, bank bases:", " ".join(f"{bo.address():#x}" for bo in bos[:4]), "...")

    def run(token, pos):
        r = krn(A["prog"], qm.n_instr, pos, 0, token, *bos)
        st = r.wait()
        return krn.read_register(0x20), krn.read_register(0x40), st

    ids = tok.chat(a.prompt)
    print(f"[host] prompt: {a.prompt!r} ({len(ids)} tokens)")
    if a.check:
        qm.reset()
    t0 = time.time()
    for pos, t in list(enumerate(ids))[1:]:
        nxt, cyc, _ = run(t, pos)
        if a.check:
            g = qm.step(t, pos)["token"]
            if g != nxt:
                print(f"\n[check] MISMATCH at prompt pos {pos}: fpga {nxt} golden {g}")
                return 1
    print(f"[host] prefill {len(ids) - 1} tokens in {time.time() - t0:.2f}s (last: {cyc} cycles = {cyc / 300:.0f} us)")
    out, t0 = [], time.time()
    sys.stdout.write("\n[FPGA] ")
    while len(out) < a.n and nxt != tok.eos:
        out.append(nxt)
        sys.stdout.write("\r[FPGA] " + tok.decode(out))
        sys.stdout.flush()
        pos = len(ids) + len(out) - 1
        prev = nxt
        nxt, cyc, _ = run(prev, pos)
        if a.check:
            g = qm.step(prev, pos)["token"]
            if g != nxt:
                print(f"\n[check] MISMATCH at gen pos {pos}: fpga {nxt} golden {g}")
                return 1
    dt = time.time() - t0
    print(f"\n[host] {len(out)} tokens in {dt:.2f}s = {len(out) / max(dt, 1e-9):.1f} tok/s "
          f"(kernel {cyc} cycles/token = {300e6 / max(cyc, 1):.0f} tok/s at 300 MHz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
