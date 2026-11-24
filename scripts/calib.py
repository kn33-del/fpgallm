import json, os, sys, time
from pathlib import Path
os.environ.setdefault("LLM_MODEL", "tinyllama")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "verif"))
from llmtb.hf import FloatLlama, Tokenizer
from llmtb.quantize import CALIB_PROMPTS, calibrate
M = str(ROOT.parent / "TinyLlama-1.1B-Chat-v1.0")
t0 = time.time()
P = calibrate(FloatLlama(M), Tokenizer(M + "/tokenizer.json"), CALIB_PROMPTS, n_new=20)
(ROOT / "build/tinyllama_calib.json").write_text(json.dumps(P))
print(f"calib done {time.time()-t0:.0f}s; x_tok percentiles:", sorted(P["x_tok"])[-12:])
