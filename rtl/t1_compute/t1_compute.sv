// T1 matrix engine. See docs/T1.md for memory packing and numerical contracts.
// The array accepts one K-slice per enabled clock. Stalls freeze data AND tags;
// operand starvation inserts invalid bubbles so outstanding results can drain.
module mm_engine import llm_pkg::*; #(
  parameter int R = ARR_R,
  parameter int C = ARR_C,
  parameter int M_MAX = MAX_SEQ,
  parameter int REQUANT_SHIFT = 16
)(
  input logic clk, rst_n,
  mm_if.slave cmd,
  mem_rd_if.client a_rd, s_rd, b_rd,
  mem_wr_if.client c_wr
);
  localparam int LATENCY = R + C - 1;
  logic [R-1:0][DATA_W-1:0] arr_a;
  logic [R-1:0][C-1:0][DATA_W-1:0] arr_w;
  logic arr_a_valid, arr_w_load, arr_stall;
  logic [C-1:0][ACC_W-1:0] psum, acc_out;
  logic psum_valid, psum_ready, acc_valid, acc_ready;
  logic first_k, last_k;
  logic [15:0] m_idx, out_m_idx, m_rows;
  logic [LATENCY-1:0] first_pipe, last_pipe;
  logic [15:0] m_pipe [LATENCY];
  logic [C-1:0][SCALE_W-1:0] scales;
  logic scales_valid;
  mm_out_e out_fmt;

  // Exactly one module owns each write-interface signal.
  mm_ctrl #(.R(R), .C(C), .M_MAX(M_MAX)) u_ctrl (
    .clk, .rst_n, .cmd, .a_rd, .s_rd, .b_rd,
    .c_req(c_wr.req), .c_req_valid(c_wr.req_valid),
    .c_req_ready(c_wr.req_ready), .c_wr_done(c_wr.done),
    .arr_a, .arr_a_valid, .arr_w, .arr_w_load,
    .arr_stall, .psum_ready,
    .acc_clear(first_k), .acc_last(last_k), .acc_m_idx(m_idx),
    .m_rows, .scales, .scales_valid, .out_fmt
  );
  systolic_array #(.R(R), .C(C)) u_arr (
    .clk, .rst_n, .a_in(arr_a), .a_valid(arr_a_valid),
    .w_in(arr_w), .w_load(arr_w_load), .stall(arr_stall),
    .psum_out(psum), .psum_valid
  );
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      first_pipe <= '0;
      last_pipe <= '0;
      for (int i = 0; i < LATENCY; i++) m_pipe[i] <= '0;
    end else if (!arr_stall) begin
      first_pipe[0] <= first_k;
      last_pipe[0] <= last_k;
      m_pipe[0] <= m_idx;
      for (int i = 1; i < LATENCY; i++) begin
        first_pipe[i] <= first_pipe[i-1];
        last_pipe[i] <= last_pipe[i-1];
        m_pipe[i] <= m_pipe[i-1];
      end
    end
  end
  mm_accum #(.C(C), .M_MAX(M_MAX)) u_acc (
    .clk, .rst_n, .psum, .psum_valid(psum_valid && !arr_stall),
    .in_ready(psum_ready), .clear(first_pipe[LATENCY-1]),
    .last(last_pipe[LATENCY-1]), .m_idx(m_pipe[LATENCY-1]),
    .out(acc_out), .out_m_idx, .out_valid(acc_valid), .out_ready(acc_ready)
  );
  mm_requant #(.C(C), .SHIFT(REQUANT_SHIFT)) u_rq (
    .clk, .rst_n, .acc(acc_out), .acc_valid, .acc_ready,
    .acc_last(out_m_idx == m_rows - 16'd1),
    .scales, .scales_valid, .fmt(out_fmt),
    .wr_data(c_wr.data), .wr_valid(c_wr.valid),
    .wr_ready(c_wr.ready), .wr_last(c_wr.last)
  );
endmodule

// A load and MAC on the same edge use the NEW weight. This bypass is essential
// for M=1: consecutive tokens may belong to different B tiles. The skew network
// stages weights ahead of their arrival at each PE; active_weight retains them
// between loads for M>1. All state freezes together during backpressure.
module pe import llm_pkg::*; (
  input logic clk, rst_n, stall,
  input logic signed [DATA_W-1:0] a_in,
  input logic a_valid_in,
  output logic signed [DATA_W-1:0] a_out,
  output logic a_valid_out,
  input logic signed [DATA_W-1:0] w_in,
  input logic w_load,
  input logic signed [ACC_W-1:0] psum_in,
  output logic signed [ACC_W-1:0] psum_out
);
  logic signed [DATA_W-1:0] active_weight, weight_operand;
  (* use_dsp = "yes" *) logic signed [2*DATA_W-1:0] product;
  assign weight_operand = w_load ? w_in : active_weight;
  assign product = a_in * weight_operand;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active_weight <= '0;
      a_out <= '0;
      a_valid_out <= 1'b0;
      psum_out <= '0;
    end else if (!stall) begin
      if (w_load) active_weight <= w_in;
      a_out <= a_in;
      a_valid_out <= a_valid_in;
      psum_out <= a_valid_in ? psum_in + ACC_W'($signed(product)) : '0;
    end
  end
endmodule

module systolic_array import llm_pkg::*; #(
  parameter int R = ARR_R,
  parameter int C = ARR_C
)(
  input logic clk, rst_n,
  input logic [R-1:0][DATA_W-1:0] a_in,
  input logic a_valid,
  input logic [R-1:0][C-1:0][DATA_W-1:0] w_in,
  input logic w_load, stall,
  output logic [C-1:0][ACC_W-1:0] psum_out,
  output logic psum_valid
);
  localparam int L = R + C - 1;
  wire [DATA_W-1:0] a_bus [R][C+1];
  wire a_v [R][C+1];
  wire [ACC_W-1:0] p_bus [R+1][C];
  logic [L-1:0] valid_pipe;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) valid_pipe <= '0;
    else if (!stall) begin
      valid_pipe[0] <= a_valid;
      for (int i = 1; i < L; i++) valid_pipe[i] <= valid_pipe[i-1];
    end
  end
  assign psum_valid = valid_pipe[L-1];

  for (genvar r = 0; r < R; r++) begin : g_row
    if (r == 0) begin : g_no_skew
      assign a_bus[r][0] = a_in[r];
      assign a_v[r][0] = a_valid;
    end else begin : g_skew
      // No reset on data delays: valid bits make their initial contents irrelevant
      // and permit SRL inference. Reset only the associated control pipeline.
      logic [DATA_W-1:0] data_pipe [r];
      logic [r-1:0] v_pipe;
      always_ff @(posedge clk) begin
        if (!stall) begin
          data_pipe[0] <= a_in[r];
          for (int i = 1; i < r; i++) data_pipe[i] <= data_pipe[i-1];
        end
      end
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_pipe <= '0;
        else if (!stall) begin
          v_pipe[0] <= a_valid;
          for (int i = 1; i < r; i++) v_pipe[i] <= v_pipe[i-1];
        end
      end
      assign a_bus[r][0] = data_pipe[r-1];
      assign a_v[r][0] = v_pipe[r-1];
    end
    for (genvar c = 0; c < C; c++) begin : g_col
      wire [DATA_W-1:0] w;
      wire load;
      if (r+c == 0) begin : g_no_delay
        assign w = w_in[r][c];
        assign load = w_load;
      end else begin : g_weight_delay
        logic [DATA_W-1:0] w_pipe [r+c];
        logic [r+c-1:0] load_pipe;
        always_ff @(posedge clk) begin
          if (!stall) begin
            w_pipe[0] <= w_in[r][c];
            for (int i = 1; i < r+c; i++) w_pipe[i] <= w_pipe[i-1];
          end
        end
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) load_pipe <= '0;
          else if (!stall) begin
            load_pipe[0] <= w_load;
            for (int i = 1; i < r+c; i++) load_pipe[i] <= load_pipe[i-1];
          end
        end
        assign w = w_pipe[r+c-1];
        assign load = load_pipe[r+c-1];
      end
      pe u_pe (
        .clk, .rst_n, .stall, .a_in(a_bus[r][c]), .a_valid_in(a_v[r][c]),
        .a_out(a_bus[r][c+1]), .a_valid_out(a_v[r][c+1]),
        .w_in(w), .w_load(load), .psum_in(p_bus[r][c]), .psum_out(p_bus[r+1][c])
      );
    end
  end
  for (genvar c = 0; c < C; c++) begin : g_output
    assign p_bus[0][c] = '0;
    if (C-1-c == 0) begin : g_no_deskew
      assign psum_out[c] = p_bus[R][c];
    end else begin : g_deskew
      logic [ACC_W-1:0] pipe_data [C-1-c];
      always_ff @(posedge clk) begin
        if (!stall) begin
          pipe_data[0] <= p_bus[R][c];
          for (int i = 1; i < C-1-c; i++) pipe_data[i] <= pipe_data[i-1];
        end
      end
      assign psum_out[c] = pipe_data[C-2-c];
    end
  end
endmodule

// Synchronous row RAM plus forwarding for consecutive updates to the same row.
// RAM is intentionally not reset: clear marks the first K tile for every row.
module mm_accum import llm_pkg::*; #(
  parameter int C = ARR_C,
  parameter int M_MAX = MAX_SEQ
)(
  input logic clk, rst_n,
  input logic [C-1:0][ACC_W-1:0] psum,
  input logic psum_valid,
  output logic in_ready,
  input logic clear, last,
  input logic [15:0] m_idx,
  output logic [C-1:0][ACC_W-1:0] out,
  output logic [15:0] out_m_idx,
  output logic out_valid,
  input logic out_ready
);
  localparam int MW = (M_MAX > 1) ? $clog2(M_MAX) : 1;
  (* ram_style = "block" *) logic [C-1:0][ACC_W-1:0] rows [M_MAX];
  logic [C-1:0][ACC_W-1:0] rd_data, stage_psum, stage_sum;
  logic stage_valid, stage_clear, stage_last;
  logic [15:0] stage_m;
  assign in_ready = !out_valid || out_ready;
  for (genvar c = 0; c < C; c++) begin : g_sum
    assign stage_sum[c] = stage_clear ? stage_psum[c] : rd_data[c] + stage_psum[c];
  end
  // Keep RAM read/write outside the reset branch for synchronous BRAM inference.
  always_ff @(posedge clk) begin
    if (rst_n && in_ready) begin
      if (stage_valid) rows[MW'(stage_m)] <= stage_sum;
      if (psum_valid) begin
        if (clear) rd_data <= '0;
        else if (stage_valid && stage_m == m_idx) rd_data <= stage_sum;
        else rd_data <= rows[MW'(m_idx)];
      end
    end
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stage_valid <= 1'b0;
      stage_clear <= 1'b0;
      stage_last <= 1'b0;
      stage_m <= '0;
      stage_psum <= '0;
      out <= '0;
      out_m_idx <= '0;
      out_valid <= 1'b0;
    end else if (in_ready) begin
      stage_valid <= psum_valid;
      if (psum_valid) begin
        stage_psum <= psum;
        stage_clear <= clear;
        stage_last <= last;
        stage_m <= m_idx;
      end
      out_valid <= stage_valid && stage_last;
      if (stage_valid && stage_last) begin
        out <= stage_sum;
        out_m_idx <= stage_m;
      end
    end
  end
`ifndef SYNTHESIS
  always_ff @(posedge clk) if (rst_n && psum_valid && in_ready)
    assert (int'(m_idx) < M_MAX) else $fatal(1, "mm_accum row out of range");
`endif
endmodule

// Two elastic stages: signed multiplier bank, then rounding/saturation and
// serializer. SHIFT is a build-time numerical contract, not an inferred format.
// Default Q16.16 is provisional until the team's golden-model format is supplied.
module mm_requant import llm_pkg::*; #(
  parameter int C = ARR_C,
  parameter int SHIFT = 16
)(
  input logic clk, rst_n,
  input logic [C-1:0][ACC_W-1:0] acc,
  input logic acc_valid,
  output logic acc_ready,
  input logic acc_last,
  input logic [C-1:0][SCALE_W-1:0] scales,
  input logic scales_valid,
  input mm_out_e fmt,
  output logic [BEAT_W-1:0] wr_data,
  output logic wr_valid,
  input logic wr_ready,
  output logic wr_last
);
  localparam int PW = ACC_W + SCALE_W;
  localparam int MAX_BEATS = (C*ACC_W + BEAT_W-1)/BEAT_W;
  localparam int BW = (MAX_BEATS > 1) ? $clog2(MAX_BEATS) : 1;
  localparam int CW = $clog2(MAX_BEATS+1);
  logic signed [PW-1:0] products [C];
  logic [C-1:0][ACC_W-1:0] raw_acc;
  logic product_valid, product_last, output_last;
  mm_out_e product_fmt;
  logic [MAX_BEATS*BEAT_W-1:0] payload;
  logic [BW-1:0] beat_idx;
  logic [CW-1:0] beat_count;
  logic output_available, product_advance, output_finish;
  logic signed [PW:0] rounded [C];
  function automatic logic signed [PW:0] round_scaled(input logic signed [PW-1:0] p);
    logic signed [PW:0] extended;
    logic [PW:0] magnitude;
    extended = {p[PW-1], p};
    magnitude = (extended < 0) ? $unsigned(-extended) : $unsigned(extended);
    if (SHIFT > 0) magnitude = (magnitude + ((PW+1)'(1) << ((SHIFT > 0) ? SHIFT-1 : 0))) >> SHIFT;
    return (extended < 0) ? -$signed(magnitude) : $signed(magnitude);
  endfunction
  for (genvar c = 0; c < C; c++) begin : g_round
    assign rounded[c] = round_scaled(products[c]);
  end
  assign output_finish = wr_valid && wr_ready && (CW'(beat_idx)+CW'(1) == beat_count);
  assign output_available = !wr_valid || output_finish;
  assign product_advance = product_valid && output_available;
  assign acc_ready = (!product_valid || product_advance) && (scales_valid || fmt == OUT_ACC32);
  assign wr_data = payload[int'(beat_idx)*BEAT_W +: BEAT_W];
  assign wr_last = wr_valid && output_last && (CW'(beat_idx)+CW'(1) == beat_count);
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      product_valid <= 1'b0;
      product_last <= 1'b0;
      product_fmt <= OUT_I8;
      raw_acc <= '0;
      for (int c = 0; c < C; c++) products[c] <= '0;
      payload <= '0;
      beat_idx <= '0;
      beat_count <= '0;
      wr_valid <= 1'b0;
      output_last <= 1'b0;
    end else begin
      if (acc_ready) begin
        product_valid <= acc_valid;
        if (acc_valid) begin
          for (int c = 0; c < C; c++) products[c] <= $signed(acc[c]) * $signed(scales[c]);
          raw_acc <= acc;
          product_fmt <= fmt;
          product_last <= acc_last;
        end
      end else if (product_advance) product_valid <= 1'b0;
      if (output_available) begin
        wr_valid <= product_valid;
        beat_idx <= '0;
        if (product_valid) begin
          payload <= '0;
          output_last <= product_last;
          case (product_fmt)
            OUT_I8: begin
              beat_count <= CW'((C*DATA_W + BEAT_W-1)/BEAT_W);
              for (int c = 0; c < C; c++) begin
                if (rounded[c] > ((PW+1)'(1) <<< (DATA_W-1))-1)
                  payload[c*DATA_W +: DATA_W] <= {1'b0, {(DATA_W-1){1'b1}}};
                else if (rounded[c] < -((PW+1)'(1) <<< (DATA_W-1)))
                  payload[c*DATA_W +: DATA_W] <= {1'b1, {(DATA_W-1){1'b0}}};
                else payload[c*DATA_W +: DATA_W] <= DATA_W'(rounded[c]);
              end
            end
            OUT_VEC: begin
              beat_count <= CW'((C*VEC_W + BEAT_W-1)/BEAT_W);
              for (int c = 0; c < C; c++) begin
                if (rounded[c] > ((PW+1)'(1) <<< (VEC_W-1))-1)
                  payload[c*VEC_W +: VEC_W] <= {1'b0, {(VEC_W-1){1'b1}}};
                else if (rounded[c] < -((PW+1)'(1) <<< (VEC_W-1)))
                  payload[c*VEC_W +: VEC_W] <= {1'b1, {(VEC_W-1){1'b0}}};
                else payload[c*VEC_W +: VEC_W] <= VEC_W'(rounded[c]);
              end
            end
            default: begin
              beat_count <= CW'(MAX_BEATS);
              for (int c = 0; c < C; c++) payload[c*ACC_W +: ACC_W] <= raw_acc[c];
            end
          endcase
        end
      end else if (wr_valid && wr_ready) beat_idx <= beat_idx + BW'(1);
    end
  end
`ifndef SYNTHESIS
  initial begin
    assert (SHIFT >= 0 && SHIFT < PW) else $fatal(1, "Invalid requant SHIFT");
    assert (VEC_W <= ACC_W && DATA_W <= ACC_W) else $fatal(1, "Invalid output width");
  end
`endif
endmodule

// n_tile -> k_tile -> m_row. Decode reads A as one contiguous stream per
// n_tile. Prefill reads one K-slice across M rows using a strided request and
// retains each B tile for those M inputs. B is one linear request per command.
module mm_ctrl import llm_pkg::*; #(
  parameter int R = ARR_R,
  parameter int C = ARR_C,
  parameter int M_MAX = MAX_SEQ
)(
  input logic clk, rst_n,
  mm_if.slave cmd,
  mem_rd_if.client a_rd, s_rd, b_rd,
  output mem_req_t c_req,
  output logic c_req_valid,
  input logic c_req_ready, c_wr_done,
  output logic [R-1:0][DATA_W-1:0] arr_a,
  output logic arr_a_valid,
  output logic [R-1:0][C-1:0][DATA_W-1:0] arr_w,
  output logic arr_w_load, arr_stall,
  input logic psum_ready,
  output logic acc_clear, acc_last,
  output logic [15:0] acc_m_idx, m_rows,
  output logic [C-1:0][SCALE_W-1:0] scales,
  output logic scales_valid,
  output mm_out_e out_fmt
);
  localparam int BEAT_BYTES = BEAT_W/8;
  localparam int TILE_BYTES = R*C*DATA_W/8;
  localparam int SCALE_BEATS = (C*SCALE_W + BEAT_W-1)/BEAT_W;
  localparam int SCW = (SCALE_BEATS > 1) ? $clog2(SCALE_BEATS) : 1;
  typedef enum logic [2:0] {IDLE, N_SETUP, A_REQ, RUN, WAIT_OUTPUT} state_e;
  state_e state;
  mm_cmd_t q;
  logic [15:0] k_tiles, n_tiles, kt, nt, mr;
  logic [31:0] a_stride, c_stride, c_tile_bytes, c_tile_beats;
  logic b_requested, b_active, a_active, s_requested, c_requested, s_loaded;
  logic write_done_seen;
  logic [SCW-1:0] scale_beat;
  logic [SCALE_BEATS*BEAT_W-1:0] scale_buffer;
  logic [BEAT_W-1:0] a_buffer;
  logic [R*C*DATA_W-1:0] b_buffer;
  logic a_full, b_full, issue, b_pop, a_take, b_take;
  // Internal counters are available to a testbench/debug probe; reset per cmd.
  logic [63:0] issued_slices, operand_wait_cycles, backpressure_cycles;

  assign cmd.cmd_ready = (state == IDLE);
  assign arr_stall = !psum_ready;
  assign issue = (state == RUN) && a_full && b_full && !arr_stall;
  assign arr_a_valid = issue;
  assign arr_w_load = issue && (mr == 0);
  assign acc_clear = (kt == 0);
  assign acc_last = (kt == k_tiles-16'd1);
  assign acc_m_idx = mr;
  assign m_rows = q.m;
  assign out_fmt = q.out_fmt;
  assign scales = scale_buffer[C*SCALE_W-1:0];
  assign scales_valid = s_loaded;
  assign b_pop = issue && (mr == q.m-16'd1);
  assign a_rd.ready = (state == RUN) && a_active && (!a_full || issue);
  assign b_rd.ready = (state != IDLE) && b_active && (!b_full || b_pop);
  assign s_rd.ready = (state == N_SETUP) && s_requested && !s_loaded;
  assign a_take = a_rd.valid && a_rd.ready;
  assign b_take = b_rd.valid && b_rd.ready;

  // Packed lane 0 is least significant. Mask both ragged K and N edges even
  // when padding bytes in memory are nonzero. Zero sums produce zero C padding.
  always_comb begin
    arr_a = '0;
    arr_w = '0;
    for (int r = 0; r < R; r++) begin
      if (int'(kt)*R+r < int'(q.k)) arr_a[r] = a_buffer[r*DATA_W +: DATA_W];
      for (int c = 0; c < C; c++) begin
        if (int'(kt)*R+r < int'(q.k) && int'(nt)*C+c < int'(q.n))
          arr_w[r][c] = b_buffer[(r*C+c)*DATA_W +: DATA_W];
      end
    end
  end
  always_comb begin
    c_tile_beats = 32'((C*DATA_W + BEAT_W-1)/BEAT_W);
    case (q.out_fmt)
      OUT_VEC: c_tile_beats = 32'((C*VEC_W + BEAT_W-1)/BEAT_W);
      OUT_ACC32: c_tile_beats = 32'((C*ACC_W + BEAT_W-1)/BEAT_W);
      default: ;
    endcase
    c_tile_bytes = c_tile_beats * BEAT_BYTES;
    c_stride = 32'(n_tiles) * c_tile_bytes;
    a_stride = 32'(k_tiles) * BEAT_BYTES;
    b_rd.req = '0;
    b_rd.req.base = q.b_base;
    b_rd.req.row_beats = k_tiles;
    b_rd.req.n_rows = n_tiles;
    b_rd.req.row_stride = (q.b_row_stride == 0)
                        ? 32'(k_tiles) * TILE_BYTES : q.b_row_stride;
    b_rd.req_valid = (state != IDLE) && !b_requested;
    s_rd.req = '0;
    s_rd.req.base = q.s_base;
    s_rd.req.base.off = q.s_base.off + AXI_AW'(int'(nt)*SCALE_BEATS*BEAT_BYTES);
    s_rd.req.row_beats = 16'(SCALE_BEATS);
    s_rd.req.n_rows = 16'd1;
    s_rd.req_valid = (state == N_SETUP) && !s_requested && q.out_fmt != OUT_ACC32;
    a_rd.req = '0;
    a_rd.req.base = q.a_base;
    if (q.m == 16'd1) begin
      a_rd.req.row_beats = k_tiles;
      a_rd.req.n_rows = 16'd1;
    end else begin
      a_rd.req.base.off = q.a_base.off + AXI_AW'(int'(kt)*BEAT_BYTES);
      a_rd.req.row_beats = 16'd1;
      a_rd.req.n_rows = q.m;
      a_rd.req.row_stride = a_stride;
    end
    a_rd.req_valid = (state == A_REQ);
    c_req = '0;
    c_req.base = q.c_base;
    c_req.base.off = q.c_base.off + AXI_AW'(32'(nt)*c_tile_bytes);
    c_req.row_beats = 16'(c_tile_beats);
    c_req.n_rows = q.m;
    c_req.row_stride = c_stride;
    c_req_valid = (state == N_SETUP) && !c_requested;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      q <= '0;
      k_tiles <= '0;
      n_tiles <= '0;
      kt <= '0;
      nt <= '0;
      mr <= '0;
      a_full <= 1'b0;
      b_full <= 1'b0;
      a_active <= 1'b0;
      b_active <= 1'b0;
      b_requested <= 1'b0;
      s_requested <= 1'b0;
      c_requested <= 1'b0;
      s_loaded <= 1'b0;
      write_done_seen <= 1'b0;
      scale_buffer <= '0;
      scale_beat <= '0;
      a_buffer <= '0;
      b_buffer <= '0;
      cmd.done <= 1'b0;
      cmd.done_tag <= '0;
      issued_slices <= '0;
      operand_wait_cycles <= '0;
      backpressure_cycles <= '0;
    end else begin
      cmd.done <= 1'b0;
      if (b_rd.req_valid && b_rd.req_ready) begin
        b_requested <= 1'b1;
        b_active <= 1'b1;
      end
      if (a_rd.req_valid && a_rd.req_ready) a_active <= 1'b1;
      if (s_rd.req_valid && s_rd.req_ready) s_requested <= 1'b1;
      if (c_req_valid && c_req_ready) c_requested <= 1'b1;
      if (c_wr_done && c_requested) write_done_seen <= 1'b1;
      if (issue) a_full <= 1'b0;
      if (b_pop) b_full <= 1'b0;
      if (a_take) begin
        a_buffer <= a_rd.data;
        a_full <= 1'b1;
        if (a_rd.last) a_active <= 1'b0;
      end
      if (b_take) begin
        b_buffer <= b_rd.data;
        b_full <= 1'b1;
        if (b_rd.last) b_active <= 1'b0;
      end
      if (s_rd.valid && s_rd.ready) begin
        scale_buffer[int'(scale_beat)*BEAT_W +: BEAT_W] <= s_rd.data;
        if (int'(scale_beat) == SCALE_BEATS-1) s_loaded <= 1'b1;
        else scale_beat <= scale_beat + SCW'(1);
      end
      if (issue) issued_slices <= issued_slices + 64'd1;
      if (state == RUN && !arr_stall && !(a_full && b_full))
        operand_wait_cycles <= operand_wait_cycles + 64'd1;
      if (state != IDLE && arr_stall) backpressure_cycles <= backpressure_cycles + 64'd1;
      case (state)
        IDLE: if (cmd.cmd_valid) begin
          q <= cmd.cmd;
          k_tiles <= 16'((32'(cmd.cmd.k)+R-1)/R);
          n_tiles <= 16'((32'(cmd.cmd.n)+C-1)/C);
          kt <= '0;
          nt <= '0;
          mr <= '0;
          a_full <= 1'b0;
          b_full <= 1'b0;
          a_active <= 1'b0;
          b_active <= 1'b0;
          b_requested <= 1'b0;
          s_requested <= 1'b0;
          c_requested <= 1'b0;
          s_loaded <= (cmd.cmd.out_fmt == OUT_ACC32);
          scale_beat <= '0;
          scale_buffer <= '0;
          write_done_seen <= 1'b0;
          issued_slices <= '0;
          operand_wait_cycles <= '0;
          backpressure_cycles <= '0;
          state <= N_SETUP;
        end
        N_SETUP: if (b_requested && c_requested && s_loaded) state <= A_REQ;
        A_REQ: if (a_rd.req_ready) state <= RUN;
        RUN: if (issue) begin
          if (mr == q.m-16'd1) begin
            mr <= '0;
            if (kt == k_tiles-16'd1) state <= WAIT_OUTPUT;
            else begin
              kt <= kt + 16'd1;
              if (q.m != 16'd1) state <= A_REQ;
            end
          end else mr <= mr + 16'd1;
        end
        WAIT_OUTPUT: if (write_done_seen || c_wr_done) begin
          if (nt == n_tiles-16'd1) begin
            cmd.done <= 1'b1;
            cmd.done_tag <= q.tag;
            state <= IDLE;
          end else begin
            nt <= nt + 16'd1;
            kt <= '0;
            mr <= '0;
            s_requested <= 1'b0;
            c_requested <= 1'b0;
            s_loaded <= (q.out_fmt == OUT_ACC32);
            scale_beat <= '0;
            scale_buffer <= '0;
            write_done_seen <= 1'b0;
            state <= N_SETUP;
          end
        end
        default: state <= IDLE;
      endcase
    end
  end
`ifndef SYNTHESIS
  initial begin
    assert (R > 0 && C > 0 && R*DATA_W <= BEAT_W && DATA_W % 8 == 0)
      else $fatal(1, "Unsupported array/read beat geometry");
    assert ($bits(b_rd.data) == R*C*DATA_W) else $fatal(1, "B port must carry one full tile");
    assert (M_MAX > 0 && M_MAX <= 65535) else $fatal(1, "Invalid M_MAX");
  end
  always_ff @(posedge clk) if (rst_n && cmd.cmd_valid && cmd.cmd_ready) begin
    assert (cmd.cmd.m > 0 && int'(cmd.cmd.m) <= M_MAX && cmd.cmd.k > 0 && cmd.cmd.n > 0)
      else $fatal(1, "Commands require 1 <= M <= M_MAX and positive K,N");
    assert (cmd.cmd.out_fmt inside {OUT_I8, OUT_VEC, OUT_ACC32})
      else $fatal(1, "Invalid output format");
  end
`endif
endmodule
