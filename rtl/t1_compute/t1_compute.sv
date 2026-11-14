// =============================================================================
// TEAM 1 -- Compute architecture. HIGHEST PRIORITY: everyone else blocks on mm_engine.
// Deliverable: verified, parameterized C = requant(A x B) engine behind mm_if.
//
// Dataflow: weight-stationary systolic array, ARR_R x ARR_C.
//   PE(r,c) holds w = B[k0+r][n0+c]. Activations enter left edge (row r gets A[m][k0+r]),
//   flow right. Partial sums flow down columns; bottom of column c = sum over r -> output n0+c.
//   Decode (M=1): every weight is used ONCE -> PEs reload weights every cycle.
//                 So the weight load path must be full-width (8192 b/cycle), shadow-buffered.
//   Prefill (M>1): weights stay M cycles, real systolic reuse. Same hardware.
// Budget: 300 MHz, 1024 MAC/cycle, fed by 280 GB/s planned -> expect ~91% busy in decode.
//   Array must NOT be the bottleneck: it must accept a B beat every cycle with zero bubbles
//   (no pipeline drain between tiles / between cmds within a layer).
// Loop order: n_tile outer, k_tile inner (matches weight layout contract) -> each column's
//   accumulator lives in mm_accum across all k_tiles, C written once per n_tile.
// =============================================================================

// -----------------------------------------------------------------------------
// mm_engine: wrapper. Wires mm_ctrl + array + accum + requant to memory ports.
// -----------------------------------------------------------------------------
module mm_engine import llm_pkg::*; (
  input logic clk, rst_n,
  mm_if.slave       cmd,
  mem_rd_if.client  a_rd,   // narrow: 1 beat = 32 int8 = A[m][k0 .. k0+31]
  mem_rd_if.client  s_rd,   // narrow: per-output-channel scales (SCALE_W each)
  mem_rd_if.client  b_rd,   // wide:   1 beat = one 32x32 int8 tile
  mem_wr_if.client  c_wr    // narrow: 32 x int8, or 32 x VEC_W (2 beats), per n_tile per m
);

  // ---- mm_ctrl -> array
  logic [ARR_R-1:0][DATA_W-1:0]        arr_a;       // one A K-slice
  logic                                arr_a_valid;
  logic [ARR_R-1:0][ARR_C-1:0][DATA_W-1:0] arr_w;   // one tile (= b_rd.data reinterpreted)
  logic                                arr_w_load;  // swap shadow -> active
  logic                                arr_stall;   // freeze whole array (operand starvation)

  // ---- array -> accum
  logic [ARR_C-1:0][ACC_W-1:0]         col_psum;
  logic                                col_psum_valid;

  // ---- mm_ctrl -> accum
  logic                                acc_clear;   // first k_tile of an n_tile
  logic                                acc_last;    // last k_tile -> emit to requant
  logic [15:0]                         acc_m_idx;   // which row of A (M>1)

  // ---- accum -> requant
  logic [ARR_C-1:0][ACC_W-1:0]         acc_out;
  logic                                acc_out_valid, acc_out_ready;

  // ---- mm_ctrl -> requant (scales for current n_tile, plus format)
  logic [ARR_C-1:0][SCALE_W-1:0]       scales;
  logic                                scales_valid;
  mm_out_e                             out_fmt;

  mm_ctrl u_ctrl (
    .clk, .rst_n,
    .cmd, .a_rd, .s_rd, .b_rd, .c_wr_done (c_wr.done),
    .arr_a, .arr_a_valid, .arr_w, .arr_w_load, .arr_stall,
    .acc_clear, .acc_last, .acc_m_idx,
    .scales, .scales_valid, .out_fmt
  );
  // mm_ctrl does:
  //   1. latch cmd; compute n_tiles = ceil(n/32), k_tiles = ceil(k/32)
  //   2. issue b_rd req (whole matrix, tiles in n-outer/k-inner order = one linear stream)
  //   3. issue s_rd req (all scales), a_rd req per n_tile (A re-read from on-chip ACT each n_tile;
  //      cheap -- 1 beat/cycle, exactly what the narrow port gives)
  //   4. issue c_wr req (whole C)
  //   5. lockstep: present one A beat + one B beat to array per cycle; stall array if either missing
  //   6. ragged edges (b_is_kv, n=pos+1): zero-pad last tile's invalid rows/cols (mask w to 0)
  //   7. pulse cmd.done after c_wr.done

  systolic_array #(.R(ARR_R), .C(ARR_C)) u_arr (
    .clk, .rst_n,
    .a_in (arr_a), .a_valid (arr_a_valid),
    .w_in (arr_w), .w_load (arr_w_load),
    .stall (arr_stall),
    .psum_out (col_psum), .psum_valid (col_psum_valid)
  );

  mm_accum #(.C(ARR_C)) u_acc (
    .clk, .rst_n,
    .psum (col_psum), .psum_valid (col_psum_valid),
    .clear (acc_clear), .last (acc_last), .m_idx (acc_m_idx),
    .out (acc_out), .out_valid (acc_out_valid), .out_ready (acc_out_ready)
  );

  mm_requant #(.C(ARR_C)) u_rq (
    .clk, .rst_n,
    .acc (acc_out), .acc_valid (acc_out_valid), .acc_ready (acc_out_ready),
    .scales, .scales_valid, .fmt (out_fmt),
    .c_wr   // drives c_wr.data/valid/last; mm_ctrl drives c_wr.req (see NOTE)
  );
  // NOTE(integration): c_wr is driven by two modules (req from mm_ctrl, data from requant).
  //   Either pass req fields through requant, or split mem_wr_if into req/data modports. Pick one.

endmodule

// -----------------------------------------------------------------------------
// pe: one INT8 MAC, weight-stationary with shadow weight register.
// -----------------------------------------------------------------------------
module pe import llm_pkg::*; (
  input  logic                     clk, rst_n,
  input  logic                     stall,
  input  logic signed [DATA_W-1:0] a_in,      // from left neighbor
  input  logic                     a_valid_in,
  output logic signed [DATA_W-1:0] a_out,     // to right neighbor (registered)
  output logic                     a_valid_out,
  input  logic signed [DATA_W-1:0] w_in,      // shadow load (already deskewed by array)
  input  logic                     w_load,    // shadow -> active
  input  logic signed [ACC_W-1:0]  psum_in,   // from PE above
  output logic signed [ACC_W-1:0]  psum_out   // to PE below = psum_in + a*w_active (registered)
);
  // TODO: MAC. Map to DSP48E2 (A*B + C, with PREG). Keep it a pure pipeline stage.
  // LATER (after functional pass): DSP INT8 packing -- 2 MACs per DSP48E2 when the two MACs
  //   share an operand (here: PE(r,c) and PE(r,c+1) share a). Halves DSP count. Xilinx WP486.
endmodule

// -----------------------------------------------------------------------------
// systolic_array: R x C grid of pe + input skew / output deskew.
// -----------------------------------------------------------------------------
module systolic_array import llm_pkg::*; #(
  parameter int R = ARR_R,
  parameter int C = ARR_C
)(
  input  logic clk, rst_n,
  input  logic [R-1:0][DATA_W-1:0]        a_in,      // unskewed K-slice
  input  logic                            a_valid,
  input  logic [R-1:0][C-1:0][DATA_W-1:0] w_in,      // unskewed tile
  input  logic                            w_load,
  input  logic                            stall,
  output logic [C-1:0][ACC_W-1:0]         psum_out,  // deskewed column sums
  output logic                            psum_valid
);
  // TODO:
  //  - a skew: row r delayed r cycles before entering column 0
  //  - w skew: PE(r,c) must swap to the new weight exactly when its new 'a' arrives = r+c cycles.
  //    Put the delay lines HERE, not in memory. In decode this is ~1024 x 8b shift regs of
  //    depth up to R+C -- watch LUT/FF cost; SRL16/SRL32 inference is your friend.
  //  - psum deskew: column c output delayed (C-1-c) so all columns align
  //  - generate R x C pe instances; psum_in of row 0 = 0
  //  - psum_valid = a_valid delayed R+C (+pipeline) cycles, respecting stall
  //  - floorplan: array must sit in SLR0 next to HBM, or the 8192-b weight bus crosses SLRs. Don't.
endmodule

// -----------------------------------------------------------------------------
// mm_accum: K-tiling accumulation. One ACC_W per column per m-row.
// -----------------------------------------------------------------------------
module mm_accum import llm_pkg::*; #(parameter int C = ARR_C) (
  input  logic clk, rst_n,
  input  logic [C-1:0][ACC_W-1:0] psum,
  input  logic                    psum_valid,
  input  logic                    clear,     // acc = psum (first k_tile)
  input  logic                    last,      // after adding, emit
  input  logic [15:0]             m_idx,     // M>1: row select into acc RAM (M=1: regs only)
  output logic [C-1:0][ACC_W-1:0] out,
  output logic                    out_valid,
  input  logic                    out_ready
);
  // TODO: decode: C registers. prefill: BRAM [M_MAX][C*ACC_W]. Back-pressure: if requant/c_wr
  //   stalls, assert ready low upstream (mm_ctrl stalls array). ACC_W=32 is enough for K<=8192 int8.
endmodule

// -----------------------------------------------------------------------------
// mm_requant: acc -> int8 or VEC_W. Bit-exact with golden model.
// -----------------------------------------------------------------------------
module mm_requant import llm_pkg::*; #(parameter int C = ARR_C) (
  input  logic clk, rst_n,
  input  logic [C-1:0][ACC_W-1:0]   acc,
  input  logic                      acc_valid,
  output logic                      acc_ready,
  input  logic [C-1:0][SCALE_W-1:0] scales,     // per output channel (already combined w*a scale)
  input  logic                      scales_valid,
  input  mm_out_e                   fmt,
  mem_wr_if.client                  c_wr
);
  // TODO: y = sat( round( (acc * mult[c]) >>> shift ) ). Pipeline the 32x32 multiply (DSP).
  //   OUT_I8: pack 32 int8 -> 1 beat. OUT_VEC: 32 x VEC_W -> 2 beats. OUT_ACC32: 4 beats.
  //   Attention scores: the 1/sqrt(HEAD_DIM) = 1/8 is folded into the scale by the host. No special case.
endmodule

// -----------------------------------------------------------------------------
// mm_ctrl: tiling sequencer + operand alignment. Ports listed; body per comments in mm_engine.
// -----------------------------------------------------------------------------
module mm_ctrl import llm_pkg::*; (
  input  logic clk, rst_n,
  mm_if.slave       cmd,
  mem_rd_if.client  a_rd,
  mem_rd_if.client  s_rd,
  mem_rd_if.client  b_rd,
  input  logic      c_wr_done,
  output logic [ARR_R-1:0][DATA_W-1:0]        arr_a,
  output logic                                arr_a_valid,
  output logic [ARR_R-1:0][ARR_C-1:0][DATA_W-1:0] arr_w,
  output logic                                arr_w_load,
  output logic                                arr_stall,
  output logic                                acc_clear,
  output logic                                acc_last,
  output logic [15:0]                         acc_m_idx,
  output logic [ARR_C-1:0][SCALE_W-1:0]       scales,
  output logic                                scales_valid,
  output mm_out_e                             out_fmt
);
  // TODO: see mm_engine comment block. Key invariant: array advances only when BOTH an A beat
  //   and a B beat are available. B (HBM) is the one that starves -- count stall cycles,
  //   that counter IS your utilization number.
endmodule
