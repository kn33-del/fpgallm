// =============================================================================
// TEAM 2 -- Attention. Deliverable: attn_unit, verified vs golden, using the SHARED mm engine.
// You own sequencing of the whole attention block, including the GEMMs (issued via mm_if).
//
// Decode at position p, one layer (all on-chip buffers in ACT/SCRATCH, KV in HBM):
//   1. GEMM  qkv = h . Wqkv            M=1, K=2048, N=3072, OUT_VEC   (fused per layout contract)
//   2. ROPE  q,k  in place             rope_unit (needs p)
//   3. KV    write k_rope, v at pos p  kv port (T4 kv_cache_mgr places it; K goes transposed)
//   4. for each Q head h (kvh = h / 4):
//        GEMM scores = q_h . K^T[kvh]  M=1, K=64,  N=p+1, b_is_kv, OUT_VEC -> SCRATCH
//        MASK  (decode: no-op, N=p+1 already causal. prefill: mask j>i)
//        SOFTMAX scores -> int8 probs  softmax_unit
//        GEMM o_h = probs . V[kvh]     M=1, K=p+1, N=64, b_is_kv, OUT_I8 -> ACT (concat heads)
//   5. GEMM  a = o . Wo                M=1, K=2048, N=2048, OUT_VEC -> out_base
//   6. pulse done
// Optimization for later: the 4 Q heads sharing a KV head = one GEMM with M=4. Free 4x on step 4.
// =============================================================================
module attn_unit import llm_pkg::*; (
  input  logic      clk, rst_n,
  input  attn_cmd_t cmd,
  input  logic      cmd_valid,
  output logic      cmd_ready,
  output logic      done,
  mm_if.master      mm,
  mem_rd_if.client  rd,   // shared by rope + softmax (only one active at a time)
  mem_wr_if.client  wr,
  mem_wr_if.client  kv
);

  // ---- seq -> rope
  logic        rope_start, rope_done;
  addr_t       rope_base;      // q|k region in ACT
  logic [15:0] rope_n_heads;   // N_HEADS for q, N_KV_HEADS for k
  logic [15:0] rope_pos;

  // ---- seq -> softmax
  logic        sm_start, sm_done;
  addr_t       sm_src, sm_dst;
  logic [15:0] sm_len;         // p+1

  // ---- seq -> kv writer
  logic        kvw_start, kvw_done;

  // ---- rd/wr port mux select (seq owns)
  localparam logic [1:0] SEL_NONE = 2'd0, SEL_ROPE = 2'd1, SEL_SM = 2'd2;
  logic [1:0] port_sel;

  attn_seq u_seq (
    .clk, .rst_n,
    .cmd, .cmd_valid, .cmd_ready, .done,
    .mm,
    .rope_start, .rope_done, .rope_base, .rope_n_heads, .rope_pos,
    .sm_start, .sm_done, .sm_src, .sm_dst, .sm_len,
    .kvw_start, .kvw_done, .kv,
    .port_sel
  );
  // attn_seq: FSM over steps 1-6 above. Builds every mm_cmd_t (addresses per layout contract).
  //   Waits for mm.done / rope_done / sm_done between steps (in-order baseline).
  //   kv write: reads k_rope + v from ACT via its own path or reuses rd port -- simplest is
  //   seq streams them: issue rd on ACT, forward beats to kv port. Decide & document.

  rope_unit u_rope (
    .clk, .rst_n,
    .start (rope_start), .done (rope_done),
    .base (rope_base), .n_heads (rope_n_heads), .pos (rope_pos)
    // rd/wr hooked through port mux below
  );

  causal_mask u_mask (
    .clk, .rst_n
    // decode: pass-through stub. Prefill: sits between score stream and softmax,
    // forces score[i][j] = -inf (VEC_W min) for j > i.
  );

  softmax_unit u_sm (
    .clk, .rst_n,
    .start (sm_start), .done (sm_done),
    .src (sm_src), .dst (sm_dst), .len (sm_len)
  );

  // TODO: port mux -- connect rd/wr to rope_unit or softmax_unit based on port_sel.
  //   (rope/softmax given internal rd/wr signals; or give them mem_rd_if ports and mux here.)

endmodule

// -----------------------------------------------------------------------------
module attn_seq import llm_pkg::*; (
  input  logic clk, rst_n,
  input  attn_cmd_t cmd, input logic cmd_valid, output logic cmd_ready, output logic done,
  mm_if.master mm,
  output logic rope_start, input logic rope_done,
  output addr_t rope_base, output logic [15:0] rope_n_heads, output logic [15:0] rope_pos,
  output logic sm_start, input logic sm_done,
  output addr_t sm_src, output addr_t sm_dst, output logic [15:0] sm_len,
  output logic kvw_start, input logic kvw_done,
  mem_wr_if.client kv,
  output logic [1:0] port_sel
);
  // TODO: FSM. Per-head loop over N_HEADS; kvh = h >> 2.
  //   K^T / V base for (layer, kvh) = per KV layout contract (T4 publishes the formula).
endmodule

// -----------------------------------------------------------------------------
// rope_unit: rotate pairs (x[2i], x[2i+1]) by theta_i * pos.
//   Llama uses split-half pairing (x[i], x[i+HEAD_DIM/2]) -- CHECK the HF impl, it bites everyone.
// -----------------------------------------------------------------------------
module rope_unit import llm_pkg::*; (
  input  logic clk, rst_n,
  input  logic start, output logic done,
  input  addr_t base, input logic [15:0] n_heads, input logic [15:0] pos
);
  // TODO: cos/sin(pos * theta_i): baseline = precomputed table [MAX_SEQ][HEAD_DIM/2] in HBM/URAM
  //   (2048*32*2*2B = 256 KB, fine). Later: CORDIC or incremental rotation.
  //   Per element pair: (x0*c - x1*s, x0*s + x1*c), VEC_W fixed point. Then requant q,k to int8
  //   (K goes into the int8 KV cache; q feeds an int8 GEMM).
endmodule

// -----------------------------------------------------------------------------
module causal_mask import llm_pkg::*; (
  input logic clk, rst_n
);
  // TODO: prefill only. Add stream in/out: score beats + (row i, col j) indices -> masked beats.
endmodule

// -----------------------------------------------------------------------------
// softmax_unit: two-pass over len scores (VEC_W) -> int8/uint8 probs.
//   pass 1: running max m and running sum s = sum exp(x - m) (online rescale when m changes)
//   pass 2: p_j = exp(x_j - m) * (1/s) -> quantize (scale 1/255, uint8 -- agree with golden model)
// -----------------------------------------------------------------------------
module softmax_unit import llm_pkg::*; (
  input  logic clk, rst_n,
  input  logic start, output logic done,
  input  addr_t src, input addr_t dst, input logic [15:0] len
);
  // TODO: exp via exp2: x*log2(e) -> int part = shift, frac part = small LUT. Reciprocal: LUT + 1 NR step.
  //   Lane-parallel (16 lanes of VEC_W per beat) for exp; reductions via adder/max tree.
endmodule
