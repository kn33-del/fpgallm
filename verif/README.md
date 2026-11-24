# pyuvm verification environment

cocotb 2.x + pyuvm + Verilator (>= 5.036). Every RTL block is checked bit-exactly
against Python reference models transcribed from the RTL numerics, and every bench
writes a profiling report.

```sh
make verif                              # all benches (run from WSL/Linux)
make verif BENCH="t1 t2"                # selected benches
python3 verif/run.py top -k TopDecode3Tokens --seed 5 --waves
```

Outputs: `build/verif/<bench>/sim.log`, `results.xml`, and `build/verif/reports/<Test>.json`.

## Layout

| Path | What it is |
| --- | --- |
| `hdl/tb_t{1..4}.sv` | Wrappers that flatten SV interfaces into plain ports (cocotb can't drive interface ports) |
| `llmtb/cfg.py` | Mirror of `llm_pkg`: params plus bit-exact packed-struct pack/unpack |
| `llmtb/golden.py` | Reference models: T1 requant GEMM, T2 RoPE/softmax, T3 RMSNorm/SiLU/mul/residual/argmax |
| `llmtb/bfm.py` | Cycle BFMs: memory servers/clients, mm engine model, 32-port AXI HBM, AXI-Lite master |
| `llmtb/uvm.py` | pyuvm layer: items, sequences, agent (sequencer + driver), scoreboard, base env/test |
| `llmtb/perf.py` | Profiling: per-port occupancy and stalls, per-GEMM array utilization and GB/s |
| `llmtb/model.py` | Host side of the full pipeline: random W8A8 model, HBM packer, program, golden decode |
| `tests/test_*.py` | Tests per bench (`t1`, `t2`, `t3`, `t4`, `top`) |

All BFMs share one clock loop. On the falling edge every component drives; 1 ns later
every component samples. A handshake is recorded iff valid and ready are both high as
the next rising edge will see them.

## Benches

- **t1** `mm_engine`: GEMV/GEMM in all output formats, ragged K/N, strided B, prefill M>1, tagged back-to-back commands, and an array-utilization profile with ideal memory.
- **t2** `attn_unit`: full decode attention for one layer at pos 0, 37, and MAX_SEQ-1, with GQA, plus contract checks of the KV-cache tag and address format. The engine is the Python reference model.
- **t3** `mlp_unit`/`vec_unit`: every vector op, edge lengths, argmax ties over the full vocab, the full SwiGLU block, and mlp/vec arbitration.
- **t4** `mem_subsys`+`ctrl_regs`: act_buf, narrow HBM (4 KB crossing, 2D), weight streamer contract layout and bandwidth, KV scatter, DMA, AXI-Lite, concurrency, and AXI error.
- **top** `llm_top` with `LLM_TINY` dims: host-driven autoregressive decode. Checks the token, final residual, logits, and HBM KV cache against the golden model every token, and profiles cycles per token and weight GB/s.

Wide (B-operand) addresses are per-PC bytes per the `llm_pkg` contract. The unit
benches store a tile contiguously at logical byte = per-PC byte * 32. The top bench
uses the real 32-PC layout.
