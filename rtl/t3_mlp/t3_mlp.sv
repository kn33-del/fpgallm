// =============================================================================
// TEAM 3 -- MLP + numerical units. Deliverable: mlp_unit (SwiGLU block, issues its own GEMMs)
// and vec_unit (RMSNorm, residual, SiLU*mul, argmax) -- the latter also serves llm_ctrl directly.
//
// MLP block, decode:
//   1. GEMM  gu = h . [Wg|Wu]      M=1, K=2048, N=16384, OUT_VEC -> SCRATCH (interleaved per n_tile)
//   2. VEC   u = quant(silu(g) .* up)  V_SILU_MUL, len = 8192      -> SCRATCH int8
//   3. GEMM  m = u . Wd            M=1, K=8192, N=2048,  OUT_VEC -> out_base
//   4. pulse mlp_done
// This block is ~2/3 of all weight bytes per layer -> it's where HBM bandwidth is measured.
// =============================================================================
module mlp_unit import llm_pkg::*; (
  input  logic        clk, rst_n,
  input  mlp_cmd_t    mlp_cmd,
  input  logic        mlp_cmd_valid,
  output logic        mlp_cmd_ready,
  output logic        mlp_done,
  input  vec_cmd_t    vec_cmd,        // direct from llm_ctrl (RMSNORM / RESADD / ARGMAX)
  input  logic        vec_cmd_valid,
  output logic        vec_cmd_ready,
  output logic        vec_done,
  output logic [31:0] argmax_idx,
  mm_if.master        mm,
  mem_rd_if.client    rd,
  mem_wr_if.client    wr
);

  // ---- mlp_seq -> vec_unit (internal SILU_MUL)
  vec_cmd_t seq_vcmd;  logic seq_vcmd_valid, seq_vcmd_ready, seq_vdone;

  // ---- muxed -> vec_unit
  vec_cmd_t vu_cmd;    logic vu_cmd_valid, vu_cmd_ready, vu_done;

  mlp_seq u_seq (
    .clk, .rst_n,
    .cmd (mlp_cmd), .cmd_valid (mlp_cmd_valid), .cmd_ready (mlp_cmd_ready), .done (mlp_done),
    .mm,
    .vcmd (seq_vcmd), .vcmd_valid (seq_vcmd_valid), .vcmd_ready (seq_vcmd_ready), .vdone (seq_vdone)
  );

  // TODO: 2:1 cmd mux into vec_unit. ctrl is in-order, so ctrl's vec_cmd and an active MLP block
  //   never overlap -> simple priority mux + route vu_done back to whoever issued.

  vec_unit u_vu (
    .clk, .rst_n,
    .cmd (vu_cmd), .cmd_valid (vu_cmd_valid), .cmd_ready (vu_cmd_ready), .done (vu_done),
    .argmax_idx,
    .rd, .wr
  );

endmodule

// -----------------------------------------------------------------------------
module mlp_seq import llm_pkg::*; (
  input  logic clk, rst_n,
  input  mlp_cmd_t cmd, input logic cmd_valid, output logic cmd_ready, output logic done,
  mm_if.master mm,
  output vec_cmd_t vcmd, output logic vcmd_valid, input logic vcmd_ready, input logic vdone
);
  // TODO: FSM steps 1-4. Addresses: Wd base = w_base + K*16384 bytes (layout contract).
endmodule

// -----------------------------------------------------------------------------
// vec_unit: streaming engine. Reads operands via rd, writes via wr, dispatches to the op datapath.
//   Lanes per beat: 16 x VEC_W in, 32 x int8 out (quantizing ops) or 16 x VEC_W out.
//   Two-operand ops (RESADD, SILU_MUL, RMSNORM gamma) share ONE rd port: alternate src0/src1 bursts
//   through a small FIFO. Vector ops are <1% of runtime; don't overbuild.
// -----------------------------------------------------------------------------
module vec_unit import llm_pkg::*; (
  input  logic clk, rst_n,
  input  vec_cmd_t cmd, input logic cmd_valid, output logic cmd_ready, output logic done,
  output logic [31:0] argmax_idx,
  mem_rd_if.client rd,
  mem_wr_if.client wr
);
  localparam int L = BEAT_W / VEC_W;   // 16 lanes

  // operand streams after rd demux
  logic [L-1:0][VEC_W-1:0] op0, op1;
  logic                    op_valid, op_ready;

  // result stream before wr
  logic [BEAT_W-1:0]       res;
  logic                    res_valid, res_ready, res_last;

  rmsnorm      #(.L(L)) u_norm (.clk, .rst_n /* , op0=x, op1=gamma, res, ctrl */);
  residual_add #(.L(L)) u_res  (.clk, .rst_n /* , op0, op1, res */);
  silu         #(.L(L)) u_silu (.clk, .rst_n /* , op0=g -> silu(g) */);
  eltwise_mul  #(.L(L)) u_mul  (.clk, .rst_n /* , silu(g), op1=up -> quant */);
  argmax       #(.L(L)) u_amax (.clk, .rst_n /* , op0=logits -> idx */);

  // TODO: cmd FSM: issue rd req(s) for src0/src1, route op stream to selected datapath,
  //   issue wr req for dst, mux selected datapath output -> wr. done after wr.done
  //   (ARGMAX: no write; done when stream drained, argmax_idx valid).
endmodule

// -----------------------------------------------------------------------------
// rmsnorm: y = x * rsqrt(mean(x^2) + eps) * gamma -> int8 (act scale from golden model)
//   Two passes over x (D_MODEL elems): pass 1 sum of squares (wide acc, ~48 b), then rsqrt
//   (LUT + 1 Newton step, or pipelined fixed-point), pass 2 multiply. x re-read from ACT, cheap.
// -----------------------------------------------------------------------------
module rmsnorm import llm_pkg::*; #(parameter int L = 16) (input logic clk, rst_n);
  // TODO
endmodule

// residual_add: y = sat(x + a), VEC_W. Trivial; it's here so the format is owned by someone.
module residual_add import llm_pkg::*; #(parameter int L = 16) (input logic clk, rst_n);
  // TODO
endmodule

// silu: x * sigmoid(x). Baseline: piecewise-linear sigmoid LUT over clamped range, VEC_W.
module silu import llm_pkg::*; #(parameter int L = 16) (input logic clk, rst_n);
  // TODO
endmodule

// eltwise_mul: y = quant_int8(a .* b). Used for SwiGLU gate * up.
module eltwise_mul import llm_pkg::*; #(parameter int L = 16) (input logic clk, rst_n);
  // TODO
endmodule

// argmax over VOCAB logits: per-beat L-way compare tree, then running max + index. Ties -> lowest idx
//   (match torch.argmax).
module argmax import llm_pkg::*; #(parameter int L = 16) (input logic clk, rst_n);
  // TODO
endmodule
