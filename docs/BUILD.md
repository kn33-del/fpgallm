# Building and rebuilding the LLM accelerator

Rule of thumb: **simulate first, synthesize only the block you changed, build the full chip rarely.**

## 1. After editing RTL: simulate (minutes)

```
python3 verif/run.py t1            # just your block (t1 | t2 | t3 | t4)
python3 verif/run.py top           # full pipeline, bit-exact vs the golden model
python3 verif/run.py               # everything (~15 min)
```

Logic bugs show up here, not in synthesis. Run it in WSL (Verilator + cocotb + pyuvm, see verif/README.md).

## 2. Size check: synthesize only your block (2-20 min, < 6 GB RAM)

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\synth_blocks.ps1 -Blocks mm_engine
```

| Team | Block        | Typical time | Peak RAM |
|------|--------------|--------------|----------|
| T1   | `mm_engine`  | ~20 min      | 5.6 GB   |
| T2   | `attn_unit`  | ~4 min       | 3.0 GB   |
| T3   | `mlp_unit`   | ~8 min       | 4.8 GB   |
| T4   | `mem_subsys` | ~10 min      | 3.3 GB   |
| ctrl | `llm_ctrl`   | ~2 min       | 2.9 GB   |

Omit `-Blocks` to run all five. Reports: `build/synth/<block>/util_synth.rpt` (and
`util_synth_hier.rpt` for a per-submodule breakdown). Default model size is `LLM_TINYLLAMA`.

**Check the log for these after every run** — each one means a memory became flip-flops,
which is what made the first full-chip synthesis take 35 GB and 10+ hours:

```
grep -E "Synth 8-(4767|11357|7186)" build/synth/<block>/vivado.log
```

How to write RAMs so Vivado maps them to BRAM/URAM:
- Element type must be a **flat vector**: `logic [C*W-1:0] mem [DEPTH];` — not `logic [C-1:0][W-1:0] mem [DEPTH];`
- The read register must be plain `if (en) q <= mem[addr];` — put clear/forwarding muxes *after* `q`.
- No reset on the RAM array or its read register (keep it out of the `negedge rst_n` block).

Note on `mem_subsys` alone: its wide weight port defaults to 256 bits when synthesized as top,
so 31 of 32 weight FIFOs are trimmed. In the full chip they add ~128 BRAM tiles.

### Vivado version
Vivado 2025.2 has **no U55C timing data** (no `xcu55c*.smax.p2p.*` speed files), so locally you
get utilization only — no timing report, checkpoint, or place & route. Use the Vivado/Vitis
release that matches the U55C platform `xilinx_u55c_gen3x16_xdma_3_202210_1` (2022.x/2023.x)
for timing and the bitstream.

### Incremental synthesis
`scripts/synth_ooc.tcl` automatically reuses `build/synth/<block>/post_synth.dcp` from the previous
run (`read_checkpoint -incremental`) when it exists. That needs a Vivado that can save the
checkpoint, i.e. the U55C-capable release above. Small edits re-synthesize much faster; big
rewrites fall back to a full run automatically. Delete the folder to force a clean run.

## 3. Bitstream (.xclbin): Linux + Vitis, hours

```
source <Vitis>/settings64.sh                 # same release as the U55C platform/XRT
./vitis/build.sh                             # RTL -> build/xo/llm_kernel.xo -> build/llm_kernel.xclbin
```

`v++` hardware linking for Alveo is Linux-only. Needs the U55C platform installed
(`platforminfo -l`). `vitis/vivado.cfg` carries the RAM size-limit setting into v++'s synthesis.
If 300 MHz does not close, v++ lowers the kernel clock automatically and reports it.

Incremental implementation for small changes after a first successful build: Vivado can reuse
the previous routed design (`INCREMENTAL_CHECKPOINT` on the implementation run, passed through
`[vivado] prop=` in a v++ config). Confirm the exact property on the Vitis release you use.

## 4. Run on the card

```
source /opt/xilinx/xrt/setup.sh
python3 host/run_llm.py --xclbin build/llm_kernel.xclbin "What is an FPGA?"
python3 host/run_llm.py --xclbin build/llm_kernel.xclbin --check     # compare each token with the CPU golden
```

Expected output (bit-exact CPU model of the datapath):
`python scripts/cpu_demo.py "What is an FPGA? Answer in one sentence."`

## Current size (TinyLlama, synthesis estimates, U55C)

| Block        | LUTs   | FFs    | BRAM36 | URAM | DSP   |
|--------------|--------|--------|--------|------|-------|
| `mm_engine`  | 32,029 | 50,918 | 57     | 0    | 1,154 |
| `mlp_unit`   | 29,359 | 19,383 | 0      | 0    | 112   |
| `mem_subsys` | 15,429 | 5,437  | 4 (+128 weight FIFOs) | 16 | 3 |
| `attn_unit`  | 2,326  | 4,438  | 1      | 0    | 7     |
| `llm_ctrl`   | 458    | 181    | 3.5    | 0    | 0     |
| **Total**    | **~80K (6%)** | **~80K (3%)** | **~194 (10%)** | **16 (2%)** | **1,276 (14%)** |

The whole design fits in one SLR (SLR0, next to HBM) with room to spare.
