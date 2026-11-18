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
module mlp_unit import llm_pkg::*; #(
  parameter int FRAC_BITS = 8,
  // Quantized outputs = sat_int8(floor(real_result * QUANT_MULT / 2**QUANT_SHIFT)).
  // Supply the golden model's inverse activation scale here.
  parameter logic [SCALE_W-1:0] QUANT_MULT = 1,
  parameter int QUANT_SHIFT = 0,
  parameter int NORM_EPS_RAW = 1,
  parameter logic [AXI_AW-1:0] SCRATCH_BASE = '0
) (
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

  mlp_seq #(.SCRATCH_BASE(SCRATCH_BASE)) u_seq (
    .clk, .rst_n,
    .cmd (mlp_cmd), .cmd_valid (mlp_cmd_valid), .cmd_ready (mlp_cmd_ready), .done (mlp_done),
    .mm,
    .vcmd (seq_vcmd), .vcmd_valid (seq_vcmd_valid), .vcmd_ready (seq_vcmd_ready), .vdone (seq_vdone)
  );

  logic vec_owner_seq;
  assign vu_cmd = seq_vcmd_valid ? seq_vcmd : vec_cmd;
  assign vu_cmd_valid = seq_vcmd_valid || vec_cmd_valid;
  assign seq_vcmd_ready = vu_cmd_ready;
  assign vec_cmd_ready = vu_cmd_ready && !seq_vcmd_valid;
  assign seq_vdone = vu_done && vec_owner_seq;
  assign vec_done = vu_done && !vec_owner_seq;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) vec_owner_seq <= 1'b0;
    else if (vu_cmd_valid && vu_cmd_ready) vec_owner_seq <= seq_vcmd_valid;
  end

  vec_unit #(.FRAC_BITS(FRAC_BITS), .QUANT_MULT(QUANT_MULT),
             .QUANT_SHIFT(QUANT_SHIFT), .NORM_EPS_RAW(NORM_EPS_RAW)) u_vu (
    .clk, .rst_n,
    .cmd (vu_cmd), .cmd_valid (vu_cmd_valid), .cmd_ready (vu_cmd_ready), .done (vu_done),
    .argmax_idx,
    .rd, .wr
  );

endmodule

// -----------------------------------------------------------------------------
module mlp_seq import llm_pkg::*; #(
  parameter logic [AXI_AW-1:0] SCRATCH_BASE = '0
) (
  input  logic clk, rst_n,
  input  mlp_cmd_t cmd, input logic cmd_valid, output logic cmd_ready, output logic done,
  mm_if.master mm,
  output vec_cmd_t vcmd, output logic vcmd_valid, input logic vcmd_ready, input logic vdone
);
  localparam int GU_BYTES = 2*D_FF*(VEC_W/8);
  // Reserve GU_BYTES + D_FF*(DATA_W/8) bytes at SCRATCH_BASE. Gate/up
  // tiles come first, then the quantized gated output. Gate/up scale arrays
  // precede Wd scales in the scale region.
  localparam int WD_OFFSET = D_MODEL*2*D_FF*(DATA_W/8);
  localparam int SD_OFFSET = 2*D_FF*(SCALE_W/8);
  localparam int UP_OFFSET = ARR_C*(VEC_W/8);
  typedef enum logic [2:0] {S_IDLE, S_GU, S_GU_WAIT, S_VEC, S_VEC_WAIT,
                            S_DOWN, S_DOWN_WAIT, S_DONE} seq_state_t;
  seq_state_t state;
  mlp_cmd_t cmd_q;
  assign cmd_ready = state == S_IDLE;
  assign done = state == S_DONE;
  assign mm.cmd_valid = state == S_GU || state == S_DOWN;
  assign vcmd_valid = state == S_VEC;
  always_comb begin
    mm.cmd = '0;
    mm.cmd.m = 16'd1;
    mm.cmd.out_fmt = OUT_VEC;
    mm.cmd.a_base = cmd_q.x_base;
    mm.cmd.b_base = cmd_q.w_base;
    mm.cmd.s_base = cmd_q.s_base;
    mm.cmd.c_base = '{rgn: RGN_SCRATCH, off: SCRATCH_BASE};
    mm.cmd.k = 16'(D_MODEL);
    mm.cmd.n = 16'(2*D_FF);
    if (state == S_DOWN || state == S_DOWN_WAIT) begin
      mm.cmd.a_base = '{rgn: RGN_SCRATCH, off: SCRATCH_BASE+AXI_AW'(GU_BYTES)};
      mm.cmd.b_base.off = cmd_q.w_base.off + AXI_AW'(WD_OFFSET);
      mm.cmd.s_base.off = cmd_q.s_base.off + AXI_AW'(SD_OFFSET);
      mm.cmd.c_base = cmd_q.out_base;
      mm.cmd.k = 16'(D_FF);
      mm.cmd.n = 16'(D_MODEL);
      mm.cmd.tag = 4'd1;
    end
    vcmd = '0;
    vcmd.op = V_SILU_MUL;
    vcmd.src0 = '{rgn: RGN_SCRATCH, off: SCRATCH_BASE};
    vcmd.src1 = '{rgn: RGN_SCRATCH, off: SCRATCH_BASE+AXI_AW'(UP_OFFSET)};
    vcmd.dst = '{rgn: RGN_SCRATCH, off: SCRATCH_BASE+AXI_AW'(GU_BYTES)};
    vcmd.len = 24'(D_FF);
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin state <= S_IDLE; cmd_q <= '0; end
    else begin
      case (state)
        S_IDLE: if (cmd_valid) begin cmd_q <= cmd; state <= S_GU; end
        S_GU: if (mm.cmd_ready) state <= S_GU_WAIT;
        S_GU_WAIT: if (mm.done) state <= S_VEC;
        S_VEC: if (vcmd_ready) state <= S_VEC_WAIT;
        S_VEC_WAIT: if (vdone) state <= S_DOWN;
        S_DOWN: if (mm.cmd_ready) state <= S_DOWN_WAIT;
        S_DOWN_WAIT: if (mm.done) state <= S_DONE;
        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule

// -----------------------------------------------------------------------------
// vec_unit: streaming engine. Reads operands via rd, writes via wr, dispatches to the op datapath.
//   Lanes per beat: 16 x VEC_W in, 32 x int8 out (quantizing ops) or 16 x VEC_W out.
//   Two-operand ops (RESADD, SILU_MUL, RMSNORM gamma) share ONE rd port: alternate src0/src1 bursts
//   through a small FIFO. Vector ops are <1% of runtime; don't overbuild.
// -----------------------------------------------------------------------------
module vec_unit import llm_pkg::*; #(
  parameter int FRAC_BITS = 8,
  parameter logic [SCALE_W-1:0] QUANT_MULT = 1,
  parameter int QUANT_SHIFT = 0,
  parameter int NORM_EPS_RAW = 1,
  parameter int CHUNK_BEATS = 16,
  parameter bit SILU_INTERLEAVED = 1'b1
) (
  input  logic clk, rst_n,
  input  vec_cmd_t cmd,
  input  logic cmd_valid,
  output logic cmd_ready,
  output logic done,
  output logic [31:0] argmax_idx,
  mem_rd_if.client rd,
  mem_wr_if.client wr
);
  localparam int L = BEAT_W / VEC_W;
  localparam int CHUNK_ELEMS = CHUNK_BEATS * L;
  localparam int IW = $clog2(CHUNK_BEATS+1);
  localparam int AW = CHUNK_BEATS > 1 ? $clog2(CHUNK_BEATS) : 1;
  localparam int VEC_BYTES = VEC_W/8;
  localparam int DATA_BYTES = DATA_W/8;
  initial begin
    assert (CHUNK_BEATS > 0 && CHUNK_BEATS <= 65535
            && (CHUNK_BEATS*L*DATA_W) % BEAT_W == 0)
      else $fatal(1, "CHUNK_BEATS must fit row_beats and pack whole output beats");
    assert (!SILU_INTERLEAVED || (ARR_C*DATA_W) % BEAT_W == 0)
      else $fatal(1, "Interleaved gate/up tiles must pack whole output beats");
  end
  typedef enum logic [3:0] {IDLE, RD0_REQ, RD0_DATA, RD1_REQ, RD1_DATA,
                            FEED, DRAIN, NORM_WAIT, WR_REQ, WR_DATA,
                            WR_WAIT, FINISH} state_t;
  state_t state;
  vec_cmd_t cmd_q;
  logic collect_norm;
  logic [23:0] position, chunk_elems;
  logic [IW-1:0] chunk_beats, read_index, feed_index, result_index, write_index;
  logic [IW-1:0] write_beats;
  logic [BEAT_W-1:0] src0_buf [CHUNK_BEATS], src1_buf [CHUNK_BEATS];
  logic [CHUNK_BEATS*BEAT_W-1:0] result_buf;
  logic [L-1:0][VEC_W-1:0] op0, op1, silu_y, residual_y;
  logic [L-1:0][DATA_W-1:0] norm_y, mul_y;
  logic op_valid, op_ready, op_last, result_valid;
  logic [$clog2(L+1)-1:0] op_count;
  logic norm_in_ready, norm_out_valid, norm_ready;
  logic residual_in_ready, residual_out_valid;
  logic silu_in_ready, silu_out_valid, mul_in_ready, mul_out_valid;
  logic arg_in_ready, arg_done, start_norm, start_arg, write_committed;
  logic [L-1:0][VEC_W-1:0] up_fifo [CHUNK_BEATS];
  logic [IW-1:0] up_write, up_read;
  logic [IW:0] up_count;
  logic accept_silu, accept_mul;
  logic [AXI_AW-1:0] read_offset;
  logic [31:0] read_element;

  // Full chunks end at output-beat boundaries; only the vector tail is padded.
  function automatic logic [23:0] next_chunk(
    input vop_e op, input logic [23:0] pos, length
  );
    logic [23:0] count, tile_left;
    tile_left = 24'(ARR_C) - (pos % 24'(ARR_C));
    count = length - pos;
    if (count > 24'(CHUNK_ELEMS)) count = 24'(CHUNK_ELEMS);
    if (op == V_SILU_MUL && SILU_INTERLEAVED && count > tile_left)
      count = tile_left;
    return count;
  endfunction

  assign cmd_ready = state == IDLE;
  assign done = state == FINISH;
  assign start_norm = cmd_valid && cmd_ready && cmd.op == V_RMSNORM;
  assign start_arg = cmd_valid && cmd_ready && cmd.op == V_ARGMAX;
  assign chunk_beats = IW'((32'(chunk_elems)+L-1)/L);
  assign write_beats = cmd_q.op == V_RESADD ? chunk_beats
                     : IW'((chunk_elems*DATA_W+BEAT_W-1)/BEAT_W);
  assign op0 = src0_buf[AW'(feed_index)];
  assign op1 = src1_buf[AW'(feed_index)];
  assign op_count = $bits(op_count)'(32'(chunk_elems)-32'(feed_index)*L >= L
                                  ? L : 32'(chunk_elems)-32'(feed_index)*L);
  assign op_last = 32'(position) + 32'(feed_index)*L + 32'(op_count) == 32'(cmd_q.len);
  assign op_valid = state == FEED;
  assign read_element = (cmd_q.op == V_SILU_MUL && SILU_INTERLEAVED)
                      ? (32'(position)/ARR_C)*(2*ARR_C)+(32'(position)%ARR_C)
                      : 32'(position);
  assign read_offset = AXI_AW'(read_element)*AXI_AW'(VEC_BYTES);

  rmsnorm #(.L(L), .FRAC_BITS(FRAC_BITS), .QUANT_MULT(QUANT_MULT),
            .QUANT_SHIFT(QUANT_SHIFT), .EPS_RAW(NORM_EPS_RAW)) u_norm (
    .clk, .rst_n, .start(start_norm), .len(cmd.len),
    .in_valid(op_valid && cmd_q.op == V_RMSNORM), .in_ready(norm_in_ready),
    .x(op0), .gamma(op1), .in_count(op_count), .in_last(op_last),
    .norm_ready, .out_valid(norm_out_valid), .out_ready(1'b1), .y(norm_y)
  );
  residual_add #(.L(L)) u_res (
    .clk, .rst_n, .in_valid(op_valid && cmd_q.op == V_RESADD),
    .in_ready(residual_in_ready), .x(op0), .a(op1),
    .out_valid(residual_out_valid), .out_ready(1'b1), .y(residual_y)
  );
  silu #(.L(L), .FRAC_BITS(FRAC_BITS)) u_silu (
    .clk, .rst_n, .in_valid(op_valid && cmd_q.op == V_SILU_MUL && up_count < (IW+1)'(CHUNK_BEATS)),
    .in_ready(silu_in_ready), .x(op0), .out_valid(silu_out_valid),
    .out_ready(mul_in_ready && up_count != 0), .y(silu_y)
  );
  eltwise_mul #(.L(L), .FRAC_BITS(FRAC_BITS), .QUANT_MULT(QUANT_MULT),
                .QUANT_SHIFT(QUANT_SHIFT)) u_mul (
    .clk, .rst_n, .in_valid(silu_out_valid && up_count != 0), .in_ready(mul_in_ready),
    .a(silu_y), .b(up_fifo[AW'(up_read)]), .out_valid(mul_out_valid), .out_ready(1'b1), .y(mul_y)
  );
  argmax #(.L(L)) u_amax (
    .clk, .rst_n, .start(start_arg), .in_valid(op_valid && cmd_q.op == V_ARGMAX),
    .in_ready(arg_in_ready), .x(op0), .in_count(op_count), .in_last(op_last),
    .done(arg_done), .argmax_idx
  );

  // Keep each up beat paired with its gate across the SiLU pipeline.
  assign accept_silu = op_valid && cmd_q.op == V_SILU_MUL && op_ready;
  assign accept_mul = silu_out_valid && mul_in_ready && up_count != 0;
  assign write_committed = wr.done && (state == WR_WAIT
                          || (state == WR_DATA && wr.ready && wr.last));
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      up_write <= '0;
      up_read <= '0;
      up_count <= '0;
    end else begin
      if ((cmd_valid && cmd_ready) || write_committed) begin
        up_write <= '0;
        up_read <= '0;
        up_count <= '0;
      end else begin
        case ({accept_silu, accept_mul})
          2'b10: up_count <= up_count + 1'b1;
          2'b01: up_count <= up_count - 1'b1;
          default: ;
        endcase
        if (accept_silu) begin
          up_fifo[AW'(up_write)] <= op1;
          up_write <= up_write == IW'(CHUNK_BEATS-1) ? '0 : up_write + 1'b1;
        end
        if (accept_mul) up_read <= up_read == IW'(CHUNK_BEATS-1) ? '0 : up_read + 1'b1;
      end
    end
  end

  always_comb begin
    op_ready = 1'b0;
    result_valid = 1'b0;
    case (cmd_q.op)
      V_RMSNORM: begin op_ready = norm_in_ready; result_valid = norm_out_valid; end
      V_RESADD: begin op_ready = residual_in_ready; result_valid = residual_out_valid; end
      V_SILU_MUL: begin
        op_ready = silu_in_ready && up_count < (IW+1)'(CHUNK_BEATS);
        result_valid = mul_out_valid;
      end
      V_ARGMAX: op_ready = arg_in_ready;
      default: ;
    endcase

    rd.req = '0;
    rd.req.base = state == RD1_REQ ? cmd_q.src1 : cmd_q.src0;
    rd.req.base.off = rd.req.base.off + read_offset;
    rd.req.row_beats = 16'(chunk_beats);
    rd.req.n_rows = 16'd1;
    rd.req_valid = state == RD0_REQ || state == RD1_REQ;
    rd.ready = state == RD0_DATA || state == RD1_DATA;

    wr.req = '0;
    wr.req.base = cmd_q.dst;
    wr.req.base.off = cmd_q.dst.off + AXI_AW'(position)
                    * (cmd_q.op == V_RESADD ? AXI_AW'(VEC_BYTES) : AXI_AW'(DATA_BYTES));
    wr.req.row_beats = 16'(write_beats);
    wr.req.n_rows = 16'd1;
    wr.req_valid = state == WR_REQ;
    wr.valid = state == WR_DATA;
    wr.data = result_buf[write_index*BEAT_W +: BEAT_W];
    wr.last = write_index + 1'b1 == write_beats;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      cmd_q <= '0;
      collect_norm <= 1'b0;
      position <= '0;
      chunk_elems <= '0;
      read_index <= '0;
      feed_index <= '0;
      result_index <= '0;
      write_index <= '0;
      result_buf <= '0;
    end else begin
      case (state)
        IDLE: if (cmd_valid) begin
          cmd_q <= cmd;
          collect_norm <= cmd.op == V_RMSNORM;
          position <= '0;
          chunk_elems <= next_chunk(cmd.op, 0, cmd.len);
          result_buf <= '0;
          result_index <= '0;
          state <= cmd.len == 0 || !(cmd.op inside {V_RMSNORM,V_RESADD,V_SILU_MUL,V_ARGMAX})
                 ? FINISH : RD0_REQ;
        end
        RD0_REQ: if (rd.req_ready) begin read_index <= '0; state <= RD0_DATA; end
        RD0_DATA: if (rd.valid) begin
          src0_buf[AW'(read_index)] <= rd.data;
          read_index <= read_index + 1'b1;
          if (rd.last) begin
            feed_index <= '0;
            state <= collect_norm || cmd_q.op == V_ARGMAX ? FEED : RD1_REQ;
          end
        end
        RD1_REQ: if (rd.req_ready) begin read_index <= '0; state <= RD1_DATA; end
        RD1_DATA: if (rd.valid) begin
          src1_buf[AW'(read_index)] <= rd.data;
          read_index <= read_index + 1'b1;
          if (rd.last) begin feed_index <= '0; state <= FEED; end
        end
        FEED: if (op_ready) begin
          feed_index <= feed_index + 1'b1;
          if (feed_index + 1'b1 == chunk_beats) begin
            if (collect_norm || cmd_q.op == V_ARGMAX) begin
              if (position + chunk_elems == cmd_q.len)
                state <= collect_norm ? NORM_WAIT : DRAIN;
              else begin
                position <= position + chunk_elems;
                chunk_elems <= next_chunk(cmd_q.op, position+chunk_elems, cmd_q.len);
                state <= RD0_REQ;
              end
            end else state <= DRAIN;
          end
        end
        DRAIN: if (cmd_q.op == V_ARGMAX && arg_done) state <= FINISH;
        NORM_WAIT: if (norm_ready) begin
          collect_norm <= 1'b0;
          position <= '0;
          chunk_elems <= next_chunk(cmd_q.op, 0, cmd_q.len);
          state <= RD0_REQ;
        end
        WR_REQ: if (wr.req_ready) begin write_index <= '0; state <= WR_DATA; end
        WR_DATA: if (wr.ready) begin
          write_index <= write_index + 1'b1;
          if (wr.last) state <= WR_WAIT;
        end
        FINISH: state <= IDLE;
        default: ;
      endcase

      if (write_committed) begin
        if (position+chunk_elems == cmd_q.len) state <= FINISH;
        else begin
          position <= position + chunk_elems;
          chunk_elems <= next_chunk(cmd_q.op, position+chunk_elems, cmd_q.len);
          result_index <= '0;
          result_buf <= '0;
          state <= RD0_REQ;
        end
      end

      if ((state == FEED || state == DRAIN) && result_valid && !collect_norm) begin
        for (int lane = 0; lane < L; lane++) begin
          if (cmd_q.op == V_RESADD)
            result_buf[result_index*BEAT_W+lane*VEC_W +: VEC_W]
              <= 32'(result_index)*L+32'(lane) < 32'(chunk_elems) ? residual_y[lane] : '0;
          else
            result_buf[result_index*L*DATA_W+lane*DATA_W +: DATA_W]
              <= 32'(result_index)*L+32'(lane) < 32'(chunk_elems)
               ? (cmd_q.op == V_RMSNORM ? norm_y[lane] : mul_y[lane]) : '0;
        end
        result_index <= result_index + 1'b1;
        if (result_index + 1'b1 == chunk_beats) state <= WR_REQ;
      end
    end
  end
endmodule

// -----------------------------------------------------------------------------
// rmsnorm: y = x * rsqrt(mean(x^2) + eps) * gamma -> int8 (act scale from golden model)
//   Two passes over x (D_MODEL elems): pass 1 sum of squares (wide acc, ~48 b), then rsqrt
//   (LUT + 1 Newton step, or pipelined fixed-point), pass 2 multiply. x re-read from ACT, cheap.
// -----------------------------------------------------------------------------
module rmsnorm import llm_pkg::*; #(
  parameter int L = 16,
  parameter int FRAC_BITS = 8,
  parameter logic [SCALE_W-1:0] QUANT_MULT = 1,
  parameter int QUANT_SHIFT = 0,
  parameter int EPS_RAW = 1  // Epsilon in Q(2*FRAC_BITS), rounded to at least one LSB.
) (
  input  logic clk, rst_n,
  input  logic start,
  input  logic [23:0] len,
  input  logic in_valid,
  output logic in_ready,
  input  logic [L-1:0][VEC_W-1:0] x, gamma,
  input  logic [$clog2(L+1)-1:0] in_count,
  input  logic in_last,
  output logic norm_ready,
  output logic out_valid,
  input  logic out_ready,
  output logic [L-1:0][DATA_W-1:0] y
);
  localparam int TREE_L = 1 << $clog2(L);
  localparam int SUM_W = 2*VEC_W + 24;
  localparam int ROOT_FRAC = 8;
  // Eight guard bits in the integer square root avoid rounding the RMS to
  // whole input LSBs. The reciprocal carries sixteen fractional bits.
  localparam int GAIN_FRAC = 16;
  localparam int ROOT_W = VEC_W + ROOT_FRAC + 1;
  localparam int GAIN_W = FRAC_BITS + GAIN_FRAC + 1;
  localparam int DIV_W = (SUM_W > GAIN_FRAC+FRAC_BITS+ROOT_FRAC+1)
                      ? SUM_W : GAIN_FRAC+FRAC_BITS+ROOT_FRAC+1;
  localparam int NORM_W = 2*VEC_W + GAIN_W + 1;
  localparam int QUANT_W = NORM_W + SCALE_W + 1;
  typedef enum logic [2:0] {N_IDLE, N_SUM, N_MEAN, N_MEAN_WAIT,
                            N_ROOT, N_ROOT_WAIT, N_GAIN, N_APPLY} norm_state_t;
  norm_state_t state;
  logic [23:0] length;
  logic [SUM_W-1:0] sum_sq, sum_tree [2*TREE_L];
  logic [2*ROOT_W-1:0] radicand;
  logic [ROOT_W-1:0] root;
  logic [GAIN_W-1:0] gain;
  logic div_start, div_done, root_done, gain_pending;
  logic [DIV_W-1:0] dividend, divisor, quotient;
  logic product_valid, norm_valid, product_ready, norm_pipe_ready, result_ready;

  for (genvar i = 0; i < TREE_L; i++) begin : g_square
    if (i < L) begin : g_active
      logic signed [2*VEC_W-1:0] square;
      assign square = $signed(x[i]) * $signed(x[i]);
      assign sum_tree[TREE_L+i] = (i < in_count) ? SUM_W'($unsigned(square)) : '0;
    end else begin : g_pad
      assign sum_tree[TREE_L+i] = '0;
    end
  end
  for (genvar i = 1; i < TREE_L; i++) begin : g_sum
    assign sum_tree[i] = sum_tree[2*i] + sum_tree[2*i+1];
  end

  assign div_start = state == N_MEAN || (state == N_GAIN && !gain_pending);
  assign dividend = state == N_MEAN ? DIV_W'(sum_sq)
                  : (DIV_W'(1) << (GAIN_FRAC+FRAC_BITS+ROOT_FRAC));
  assign divisor = state == N_MEAN ? DIV_W'(length) : DIV_W'(root);
  t3_unsigned_div #(.W(DIV_W)) u_div (
    .clk, .rst_n, .start(div_start), .dividend, .divisor, .done(div_done), .quotient
  );
  t3_isqrt #(.ROOT_W(ROOT_W)) u_root (
    .clk, .rst_n, .start(state == N_ROOT), .radicand, .done(root_done), .root
  );

  assign norm_ready = state == N_APPLY;
  assign result_ready = !out_valid || out_ready;
  assign norm_pipe_ready = !norm_valid || result_ready;
  assign product_ready = !product_valid || norm_pipe_ready;
  assign in_ready = state == N_SUM || (state == N_APPLY && product_ready);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= N_IDLE;
      length <= '0;
      sum_sq <= '0;
      radicand <= '0;
      gain <= '0;
      gain_pending <= 1'b0;
    end else if (start) begin
      length <= len;
      sum_sq <= '0;
      gain_pending <= 1'b0;
      state <= len == 0 ? N_APPLY : N_SUM;
    end else begin
      case (state)
        N_SUM: if (in_valid && in_ready) begin
          sum_sq <= sum_sq + sum_tree[1];
          if (in_last) state <= N_MEAN;
        end
        N_MEAN: state <= N_MEAN_WAIT;
        N_MEAN_WAIT: if (div_done) begin
          radicand <= ((2*ROOT_W)'(quotient) + (2*ROOT_W)'(EPS_RAW > 0 ? EPS_RAW : 1))
                    << (2*ROOT_FRAC);
          state <= N_ROOT;
        end
        N_ROOT: state <= N_ROOT_WAIT;
        N_ROOT_WAIT: if (root_done) state <= N_GAIN;
        N_GAIN: begin
          gain_pending <= 1'b1;
          if (div_done) begin
            gain <= GAIN_W'(quotient);
            state <= N_APPLY;
          end
        end
        default: ;
      endcase
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      product_valid <= 1'b0;
      norm_valid <= 1'b0;
      out_valid <= 1'b0;
    end else if (start) begin
      product_valid <= 1'b0;
      norm_valid <= 1'b0;
      out_valid <= 1'b0;
    end else begin
      if (product_ready) product_valid <= in_valid && state == N_APPLY;
      if (norm_pipe_ready) norm_valid <= product_valid;
      if (result_ready) out_valid <= norm_valid;
    end
  end

  for (genvar lane = 0; lane < L; lane++) begin : g_lane
    logic signed [2*VEC_W-1:0] x_gamma;
    logic signed [NORM_W-1:0] normalized;
    logic signed [QUANT_W-1:0] scaled, quantized;
    assign scaled = normalized * $signed({1'b0, QUANT_MULT});
    assign quantized = scaled >>> (2*FRAC_BITS+GAIN_FRAC+QUANT_SHIFT);
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        x_gamma <= '0;
        normalized <= '0;
        y[lane] <= '0;
      end else begin
        if (state == N_APPLY && in_valid && product_ready)
          x_gamma <= $signed(x[lane]) * $signed(gamma[lane]);
        if (norm_pipe_ready && product_valid)
          normalized <= x_gamma * $signed({1'b0, gain});
        if (result_ready && norm_valid) begin
          if (quantized > (QUANT_W'(1) << (DATA_W-1))-1)
            y[lane] <= {1'b0, {(DATA_W-1){1'b1}}};
          else if (quantized < -(QUANT_W'(1) << (DATA_W-1)))
            y[lane] <= {1'b1, {(DATA_W-1){1'b0}}};
          else y[lane] <= DATA_W'(quantized);
        end
      end
    end
  end
endmodule

// residual_add: y = sat(x + a), VEC_W. Trivial; it's here so the format is owned by someone.
module residual_add import llm_pkg::*; #(parameter int L = 16) (
  input  logic clk, rst_n,
  input  logic in_valid,
  output logic in_ready,
  input  logic [L-1:0][VEC_W-1:0] x, a,
  output logic out_valid,
  input  logic out_ready,
  output logic [L-1:0][VEC_W-1:0] y
);
  logic [L-1:0][VEC_W-1:0] saturated_sum;

  for (genvar lane = 0; lane < L; lane++) begin : g_lane
    logic signed [VEC_W:0] sum;
    assign sum = $signed({x[lane][VEC_W-1], x[lane]})
               + $signed({a[lane][VEC_W-1], a[lane]});
    assign saturated_sum[lane] = (sum[VEC_W] != sum[VEC_W-1])
                              ? {sum[VEC_W], {(VEC_W-1){~sum[VEC_W]}}}
                              : sum[VEC_W-1:0];
  end

  assign in_ready = !out_valid || out_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      y <= '0;
    end else if (in_ready) begin
      out_valid <= in_valid;
      if (in_valid)
        y <= saturated_sum;
    end
  end
endmodule

// silu: x * sigmoid(x). Baseline: piecewise-linear sigmoid LUT over clamped range, VEC_W.
module silu import llm_pkg::*; #(
  parameter int L = 16,
  parameter int FRAC_BITS = 8  // Input/output fractional bits; 1 <= FRAC_BITS < VEC_W.
) (
  input  logic clk, rst_n,
  input  logic in_valid,
  output logic in_ready,
  input  logic [L-1:0][VEC_W-1:0] x,
  output logic out_valid,
  input  logic out_ready,
  output logic [L-1:0][VEC_W-1:0] y
);
  localparam int SIG_FRAC = 16;
  logic lookup_valid, interp_valid;
  logic lookup_ready, interp_ready, result_ready;

  // Positive sigmoid samples at half-unit intervals, Q0.16 base and delta.
  function automatic logic [30:0] sigmoid_segment(input logic [3:0] index);
    case (index)
      4'd0: sigmoid_segment = {17'd32768, 14'd8025};
      4'd1: sigmoid_segment = {17'd40793, 14'd7118};
      4'd2: sigmoid_segment = {17'd47911, 14'd5670};
      4'd3: sigmoid_segment = {17'd53581, 14'd4143};
      4'd4: sigmoid_segment = {17'd57724, 14'd2841};
      4'd5: sigmoid_segment = {17'd60565, 14'd1863};
      4'd6: sigmoid_segment = {17'd62428, 14'd1187};
      4'd7: sigmoid_segment = {17'd63615, 14'd742};
      4'd8: sigmoid_segment = {17'd64357, 14'd459};
      4'd9: sigmoid_segment = {17'd64816, 14'd281};
      4'd10: sigmoid_segment = {17'd65097, 14'd172};
      4'd11: sigmoid_segment = {17'd65269, 14'd105};
      4'd12: sigmoid_segment = {17'd65374, 14'd64};
      4'd13: sigmoid_segment = {17'd65438, 14'd38};
      4'd14: sigmoid_segment = {17'd65476, 14'd24};
      4'd15: sigmoid_segment = {17'd65500, 14'd14};
      default: sigmoid_segment = '0;
    endcase
  endfunction

  assign result_ready = !out_valid || out_ready;
  assign interp_ready = !interp_valid || result_ready;
  assign lookup_ready = !lookup_valid || interp_ready;
  assign in_ready = lookup_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lookup_valid <= 1'b0;
      interp_valid <= 1'b0;
      out_valid <= 1'b0;
    end else begin
      if (lookup_ready) lookup_valid <= in_valid;
      if (interp_ready) interp_valid <= lookup_valid;
      if (result_ready) out_valid <= interp_valid;
    end
  end

  for (genvar lane = 0; lane < L; lane++) begin : g_lane
    logic [VEC_W:0] magnitude, segment;
    logic [FRAC_BITS-1:0] fraction;
    logic [16:0] base_next, base_reg, positive_sigmoid, sigmoid_reg;
    logic [13:0] delta_next, delta_reg;
    logic [FRAC_BITS-1:0] fraction_reg;
    logic [FRAC_BITS+13:0] interpolation;
    logic signed [VEC_W-1:0] lookup_x, interp_x;
    logic signed [VEC_W+17:0] product;

    // Widen before negation so the most-negative input is handled correctly.
    assign magnitude = x[lane][VEC_W-1]
                     ? -$signed({x[lane][VEC_W-1], x[lane]})
                     : $signed({1'b0, x[lane]});
    assign segment = magnitude >> (FRAC_BITS-1);
    assign fraction = magnitude[FRAC_BITS-1:0]
                    & {1'b0, {(FRAC_BITS-1){1'b1}}};
    // Clamp sigmoid to zero/one outside [-8, 8].
    assign {base_next, delta_next} = (segment >= 16)
                                  ? {17'd65536, 14'd0}
                                  : sigmoid_segment(segment[3:0]);

    assign interpolation = delta_reg * fraction_reg;
    assign positive_sigmoid = base_reg
                            + 17'(interpolation >> (FRAC_BITS-1));
    assign product = interp_x * $signed({1'b0, sigmoid_reg});

    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        lookup_x <= '0;
        base_reg <= '0;
        delta_reg <= '0;
        fraction_reg <= '0;
        interp_x <= '0;
        sigmoid_reg <= '0;
        y[lane] <= '0;
      end else begin
        if (lookup_ready && in_valid) begin
          lookup_x <= $signed(x[lane]);
          base_reg <= base_next;
          delta_reg <= delta_next;
          fraction_reg <= fraction;
        end
        if (interp_ready && lookup_valid) begin
          interp_x <= lookup_x;
          sigmoid_reg <= lookup_x[VEC_W-1]
                       ? 17'd65536 - positive_sigmoid : positive_sigmoid;
        end
        if (result_ready && interp_valid)
          // Arithmetic truncation back to the input/output Q point.
          y[lane] <= product[SIG_FRAC +: VEC_W];
      end
    end
  end
endmodule

// eltwise_mul: y = sat_int8((a*b*QUANT_MULT) >>> (2*FRAC_BITS+QUANT_SHIFT)).
// Used for SwiGLU gate * up; negative results use arithmetic floor truncation.
module eltwise_mul import llm_pkg::*; #(
  parameter int L = 16,
  parameter int FRAC_BITS = 8,
  parameter logic [SCALE_W-1:0] QUANT_MULT = 1,
  parameter int QUANT_SHIFT = 0
) (
  input logic clk, rst_n,
  input logic in_valid,
  output logic in_ready,
  input logic [L-1:0][VEC_W-1:0] a, b,
  output logic out_valid,
  input logic out_ready,
  output logic [L-1:0][DATA_W-1:0] y
);
  localparam int QW = 2*VEC_W + SCALE_W + 1;
  logic product_valid, product_ready, result_ready;
  assign result_ready = !out_valid || out_ready;
  assign product_ready = !product_valid || result_ready;
  assign in_ready = product_ready;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      product_valid <= 1'b0;
      out_valid <= 1'b0;
    end else begin
      if (product_ready) product_valid <= in_valid;
      if (result_ready) out_valid <= product_valid;
    end
  end
  for (genvar lane = 0; lane < L; lane++) begin : g_lane
    logic signed [2*VEC_W-1:0] product;
    logic signed [QW-1:0] scaled, quantized;
    assign scaled = product * $signed({1'b0, QUANT_MULT});
    assign quantized = scaled >>> (2*FRAC_BITS+QUANT_SHIFT);
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        product <= '0;
        y[lane] <= '0;
      end else begin
        if (product_ready && in_valid)
          product <= $signed(a[lane]) * $signed(b[lane]);
        if (result_ready && product_valid) begin
          if (quantized > (QW'(1) << (DATA_W-1))-1)
            y[lane] <= {1'b0, {(DATA_W-1){1'b1}}};
          else if (quantized < -(QW'(1) << (DATA_W-1)))
            y[lane] <= {1'b1, {(DATA_W-1){1'b0}}};
          else y[lane] <= DATA_W'(quantized);
        end
      end
    end
  end
endmodule

// argmax over VOCAB logits: per-beat L-way compare tree, then running max + index. Ties -> lowest idx
//   (match torch.argmax).
module argmax import llm_pkg::*; #(parameter int L = 16) (
  input logic clk, rst_n,
  input logic start,
  input logic in_valid,
  output logic in_ready,
  input logic [L-1:0][VEC_W-1:0] x,
  input logic [$clog2(L+1)-1:0] in_count,
  input logic in_last,
  output logic done,
  output logic [31:0] argmax_idx
);
  localparam int TREE_L = 1 << $clog2(L);
  logic signed [VEC_W-1:0] value [2*TREE_L];
  logic [31:0] index [2*TREE_L];
  logic valid [2*TREE_L];
  logic have_max;
  logic signed [VEC_W-1:0] running_max;
  logic [31:0] offset;
  assign in_ready = 1'b1;
  for (genvar i = 0; i < TREE_L; i++) begin : g_leaf
    if (i < L) begin : g_active
      assign value[TREE_L+i] = $signed(x[i]);
      assign valid[TREE_L+i] = i < in_count;
    end else begin : g_pad
      assign value[TREE_L+i] = '0;
      assign valid[TREE_L+i] = 1'b0;
    end
    assign index[TREE_L+i] = 32'(i);
  end
  for (genvar i = 1; i < TREE_L; i++) begin : g_compare
    logic take_left;
    assign take_left = valid[2*i] && (!valid[2*i+1] || value[2*i] >= value[2*i+1]);
    assign value[i] = take_left ? value[2*i] : value[2*i+1];
    assign index[i] = take_left ? index[2*i] : index[2*i+1];
    assign valid[i] = valid[2*i] || valid[2*i+1];
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      have_max <= 1'b0;
      running_max <= '0;
      offset <= '0;
      argmax_idx <= '0;
      done <= 1'b0;
    end else begin
      done <= 1'b0;
      if (start) begin
        have_max <= 1'b0;
        offset <= '0;
        argmax_idx <= '0;
      end else if (in_valid && in_ready) begin
        if (valid[1] && (!have_max || value[1] > running_max)) begin
          running_max <= value[1];
          argmax_idx <= offset + index[1];
          have_max <= 1'b1;
        end
        offset <= offset + 32'(in_count);
        done <= in_last;
      end
    end
  end
endmodule

// Restoring division and digit-by-digit square root keep scalar normalization
// setup off the streaming critical path. Each consumes one bit/digit per cycle.
module t3_unsigned_div #(parameter int W = 56) (
  input logic clk, rst_n, start,
  input logic [W-1:0] dividend, divisor,
  output logic done,
  output logic [W-1:0] quotient
);
  logic busy;
  logic [W-1:0] divisor_q, dividend_q;
  logic [W:0] remainder_q, shifted;
  logic [$clog2(W+1)-1:0] count;
  assign shifted = {remainder_q[W-1:0], dividend_q[W-1]};
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0;
      done <= 1'b0;
      quotient <= '0;
      divisor_q <= '0;
      dividend_q <= '0;
      remainder_q <= '0;
      count <= '0;
    end else begin
      done <= 1'b0;
      if (start && !busy) begin
        divisor_q <= divisor;
        dividend_q <= dividend;
        remainder_q <= '0;
        quotient <= '0;
        count <= $bits(count)'(W);
        busy <= 1'b1;
      end else if (busy) begin
        dividend_q <= dividend_q << 1;
        if (shifted >= {1'b0, divisor_q}) begin
          remainder_q <= shifted - {1'b0, divisor_q};
          quotient <= (quotient << 1) | W'(1);
        end else begin
          remainder_q <= shifted;
          quotient <= quotient << 1;
        end
        count <= count - 1'b1;
        if (count == 1) begin
          busy <= 1'b0;
          done <= 1'b1;
        end
      end
    end
  end
endmodule

module t3_isqrt #(parameter int ROOT_W = 25) (
  input logic clk, rst_n, start,
  input logic [2*ROOT_W-1:0] radicand,
  output logic done,
  output logic [ROOT_W-1:0] root
);
  logic busy;
  logic [2*ROOT_W-1:0] radicand_q;
  logic [ROOT_W+1:0] remainder_q, shifted, trial;
  logic [$clog2(ROOT_W+1)-1:0] count;
  assign shifted = (remainder_q << 2) | (ROOT_W+2)'(radicand_q[2*ROOT_W-1 -: 2]);
  assign trial = {root, 2'b01};
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0;
      done <= 1'b0;
      root <= '0;
      radicand_q <= '0;
      remainder_q <= '0;
      count <= '0;
    end else begin
      done <= 1'b0;
      if (start && !busy) begin
        radicand_q <= radicand;
        remainder_q <= '0;
        root <= '0;
        count <= $bits(count)'(ROOT_W);
        busy <= 1'b1;
      end else if (busy) begin
        radicand_q <= radicand_q << 2;
        if (shifted >= trial) begin
          remainder_q <= shifted - trial;
          root <= (root << 1) | ROOT_W'(1);
        end else begin
          remainder_q <= shifted;
          root <= root << 1;
        end
        count <= count - 1'b1;
        if (count == 1) begin
          busy <= 1'b0;
          done <= 1'b1;
        end
      end
    end
  end
endmodule
