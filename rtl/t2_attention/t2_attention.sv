// Decode-only grouped-query attention. See docs/T2.md for the memory and
// numerical contracts shared with the matrix engine and weight packer.
package t2_attn_pkg;
  import llm_pkg::*;

  localparam int BEAT_BYTES = BEAT_W / 8;
  localparam int HEAD_BYTES = HEAD_DIM * DATA_W / 8;
  localparam int HEAD_BEATS = HEAD_BYTES / BEAT_BYTES;
  localparam int HEAD_VEC_BYTES = HEAD_DIM * VEC_W / 8;
  localparam int QKV_ELEMENTS = (N_HEADS + 2*N_KV_HEADS) * HEAD_DIM;
  localparam int QKV_VEC_BYTES = QKV_ELEMENTS * VEC_W / 8;
  localparam int QKV_BYTES = QKV_ELEMENTS * DATA_W / 8;
  localparam int SCORE_BYTES = ((MAX_SEQ + ARR_C-1) / ARR_C) * ARR_C * VEC_W / 8;
  localparam int PROB_BYTES = ((MAX_SEQ + ARR_R-1) / ARR_R) * BEAT_BYTES;
  localparam int SCRATCH_BYTES = QKV_VEC_BYTES + QKV_BYTES + SCORE_BYTES
                                 + PROB_BYTES + D_MODEL * DATA_W / 8;
  localparam int COEFF_BYTES = HEAD_DIM * 16 / 8;
  localparam int TILE_BYTES = ARR_R * ARR_C * DATA_W / 8;
  // Wide (B-operand) addresses are PER-PC bytes per the llm_pkg contract: one tile = 32 B in
  // each of the N_HBM pseudo-channels.
  localparam int TILE_PC_BYTES = TILE_BYTES / N_HBM;
  localparam int WO_OFFSET = ((D_MODEL + ARR_R-1) / ARR_R)
                             * ((QKV_ELEMENTS + ARR_C-1) / ARR_C) * TILE_PC_BYTES;
  localparam int SCALE_TILE_BYTES = ((ARR_C*SCALE_W + BEAT_W-1) / BEAT_W) * BEAT_BYTES;
  localparam int WO_SCALE_OFFSET = ((QKV_ELEMENTS + ARR_C-1) / ARR_C) * SCALE_TILE_BYTES;
  // V cache: per-PC stride between its N tiles (k_tiles = MAX_SEQ/32 tiles each).
  localparam int V_N_TILE_STRIDE = ((MAX_SEQ + ARR_R-1) / ARR_R) * TILE_PC_BYTES;
  localparam int K_SCALE_BYTES = ((MAX_SEQ + ARR_C-1) / ARR_C) * SCALE_TILE_BYTES;
  localparam int V_SCALE_BYTES = ((HEAD_DIM + ARR_C-1) / ARR_C) * SCALE_TILE_BYTES;
  localparam int KV_SCALE_BYTES = K_SCALE_BYTES + V_SCALE_BYTES;

  // Projection outputs use seven fraction bits in units of the final int8
  // code. For example, projected code 3.5 arrives as 448 and rounds to 4.
  localparam int QKV_FRAC = 7;
  localparam int SCORE_FRAC = 8;
  localparam int COEFF_FRAC = 15;
  typedef enum logic [1:0] {PORT_IDLE, PORT_ROPE, PORT_SOFTMAX, PORT_KV} port_e;

  function automatic addr_t offset_addr(input addr_t base, input int unsigned byte_offset);
    return '{rgn: base.rgn, off: base.off + AXI_AW'(byte_offset)};
  endfunction

  function automatic mem_req_t linear_req(input addr_t base, input int unsigned beats);
    return '{base: base, row_beats: 16'(beats), n_rows: 16'd1, row_stride: 32'd0};
  endfunction
endpackage

module attn_unit
  import llm_pkg::*;
  import t2_attn_pkg::*;
#(
  parameter logic [AXI_AW-1:0] SCRATCH_BASE   = '0,
  // KV cache placement is fixed by llm_pkg::kv_pc_off (T4 kv_cache_mgr writes it there).
  parameter logic [AXI_AW-1:0] KV_SCALE_BASE  = 34'h084000000,
  parameter logic [AXI_AW-1:0] ROPE_BASE      = 34'h085000000
)(
  input logic clk, rst_n,
  input attn_cmd_t cmd,
  input logic cmd_valid,
  output logic cmd_ready, done,
  mm_if.master mm,
  mem_rd_if.client rd,
  mem_wr_if.client wr, kv
);
  localparam addr_t QKV_VEC = '{rgn: RGN_SCRATCH, off: SCRATCH_BASE};
  localparam addr_t QKV_I8 = offset_addr(QKV_VEC, QKV_VEC_BYTES);
  localparam addr_t SCORES = offset_addr(QKV_I8, QKV_BYTES);
  localparam addr_t PROBS = offset_addr(SCORES, SCORE_BYTES);
  localparam addr_t HEADS = offset_addr(PROBS, PROB_BYTES);

  port_e port_sel;
  logic rope_start, rope_done, sm_start, sm_done;
  logic [15:0] token_pos;
  mem_rd_if rope_rd(), sm_rd(), kv_src();
  mem_wr_if rope_wr(), sm_wr();

  attn_seq #(.QKV_VEC(QKV_VEC), .QKV_I8(QKV_I8), .SCORES(SCORES),
             .PROBS(PROBS), .HEADS(HEADS),
             .KV_SCALE_BASE(KV_SCALE_BASE)) u_seq (
    .clk, .rst_n, .cmd, .cmd_valid, .cmd_ready, .done, .mm,
    .token_pos, .rope_start, .rope_done, .sm_start, .sm_done,
    .kv_src, .kv, .port_sel
  );

  rope_unit #(.SRC(QKV_VEC), .DST(QKV_I8), .ROPE_BASE(ROPE_BASE)) u_rope (
    .clk, .rst_n, .start(rope_start), .done(rope_done), .pos(token_pos),
    .rd(rope_rd), .wr(rope_wr)
  );

  // The score GEMM and softmax both stop at pos+1, so decode is already causal.
  softmax_unit #(.SRC(SCORES), .DST(PROBS)) u_sm (
    .clk, .rst_n, .start(sm_start), .done(sm_done), .len(token_pos + 16'd1),
    .rd(sm_rd), .wr(sm_wr)
  );

  always_comb begin
    rd.req = '0;
    rd.req_valid = 1'b0;
    rd.ready = 1'b0;
    wr.req = '0;
    wr.req_valid = 1'b0;
    wr.data = '0;
    wr.valid = 1'b0;
    wr.last = 1'b0;
    case (port_sel)
      PORT_ROPE: begin
        rd.req = rope_rd.req;
        rd.req_valid = rope_rd.req_valid;
        rd.ready = rope_rd.ready;
        wr.req = rope_wr.req;
        wr.req_valid = rope_wr.req_valid;
        wr.data = rope_wr.data;
        wr.valid = rope_wr.valid;
        wr.last = rope_wr.last;
      end
      PORT_SOFTMAX: begin
        rd.req = sm_rd.req;
        rd.req_valid = sm_rd.req_valid;
        rd.ready = sm_rd.ready;
        wr.req = sm_wr.req;
        wr.req_valid = sm_wr.req_valid;
        wr.data = sm_wr.data;
        wr.valid = sm_wr.valid;
        wr.last = sm_wr.last;
      end
      PORT_KV: begin
        rd.req = kv_src.req;
        rd.req_valid = kv_src.req_valid;
        rd.ready = kv_src.ready;
      end
      default: ;
    endcase
  end

  assign rope_rd.req_ready = rd.req_ready && (port_sel == PORT_ROPE);
  assign rope_rd.data = rd.data;
  assign rope_rd.valid = rd.valid && (port_sel == PORT_ROPE);
  assign rope_rd.last = rd.last;
  assign rope_wr.req_ready = wr.req_ready && (port_sel == PORT_ROPE);
  assign rope_wr.ready = wr.ready && (port_sel == PORT_ROPE);
  assign rope_wr.done = wr.done && (port_sel == PORT_ROPE);

  assign sm_rd.req_ready = rd.req_ready && (port_sel == PORT_SOFTMAX);
  assign sm_rd.data = rd.data;
  assign sm_rd.valid = rd.valid && (port_sel == PORT_SOFTMAX);
  assign sm_rd.last = rd.last;
  assign sm_wr.req_ready = wr.req_ready && (port_sel == PORT_SOFTMAX);
  assign sm_wr.ready = wr.ready && (port_sel == PORT_SOFTMAX);
  assign sm_wr.done = wr.done && (port_sel == PORT_SOFTMAX);

  assign kv_src.req_ready = rd.req_ready && (port_sel == PORT_KV);
  assign kv_src.data = rd.data;
  assign kv_src.valid = rd.valid && (port_sel == PORT_KV);
  assign kv_src.last = rd.last;

`ifndef SYNTHESIS
  initial begin
    assert (D_MODEL == N_HEADS*HEAD_DIM && N_HEADS % N_KV_HEADS == 0
            && HEAD_DIM % 2 == 0 && VEC_W == 16 && DATA_W == 8 && MAX_SEQ < 65535)
      else $fatal(1, "Unsupported attention dimensions or element widths");
    assert (ARR_R*DATA_W == BEAT_W && ARR_C == ARR_R
            && HEAD_DIM % ARR_C == 0 && HEAD_BYTES % BEAT_BYTES == 0
            && COEFF_BYTES % BEAT_BYTES == 0)
      else $fatal(1, "Attention requires beat-aligned heads and T1 tiles");
    assert (SCRATCH_BASE % AXI_AW'(BEAT_BYTES) == 0
            && KV_SCALE_BASE % AXI_AW'(BEAT_BYTES) == 0
            && ROPE_BASE % AXI_AW'(BEAT_BYTES) == 0)
      else $fatal(1, "Unaligned attention memory base");
  end
`endif
endmodule

module attn_seq
  import llm_pkg::*;
  import t2_attn_pkg::*;
#(
  parameter addr_t QKV_VEC = '0, QKV_I8 = '0, SCORES = '0, PROBS = '0, HEADS = '0,
  parameter logic [AXI_AW-1:0] KV_SCALE_BASE = '0
)(
  input logic clk, rst_n,
  input attn_cmd_t cmd,
  input logic cmd_valid,
  output logic cmd_ready, done,
  mm_if.master mm,
  output logic [15:0] token_pos,
  output logic rope_start, sm_start,
  input logic rope_done, sm_done,
  mem_rd_if.client kv_src,
  mem_wr_if.client kv,
  output port_e port_sel
);
  typedef enum logic [4:0] {
    IDLE, PROJECT_QKV, WAIT_QKV, START_ROPE, WAIT_ROPE,
    KV_READ_REQ, KV_READ, KV_WRITE_REQ, KV_WRITE, KV_COMMIT,
    SCORE_MATMUL, WAIT_SCORES, START_SOFTMAX, WAIT_SOFTMAX,
    VALUE_MATMUL, WAIT_VALUES, PROJECT_OUTPUT, WAIT_OUTPUT, FINISH
  } state_e;
  state_e state;
  attn_cmd_t active;
  localparam int HEAD_INDEX_W = (N_HEADS > 1) ? $clog2(N_HEADS) : 1;
  localparam int KV_BEATS = 2*HEAD_BEATS;
  localparam int KV_BEAT_W = $clog2(KV_BEATS);
  logic [HEAD_INDEX_W-1:0] head;
  logic [KV_BEAT_W-1:0] kv_beat;
  logic [KV_BEATS*BEAT_W-1:0] kv_payload;
  logic kv_committed;
  logic kv_is_v;           // append phase: 0 = K vector request, 1 = V vector request
  int unsigned cache_head, pair_index;
  addr_t k_cache, v_cache, k_scales, v_scales;
  kv_tag_t kv_tag;

  assign cmd_ready = (state == IDLE);
  assign done = (state == FINISH);
  assign token_pos = active.pos;
  assign rope_start = (state == START_ROPE);
  assign sm_start = (state == START_SOFTMAX);

  always_comb begin
    cache_head = int'(head);
    if (state == SCORE_MATMUL || state == VALUE_MATMUL)
      cache_head = int'(head) / (N_HEADS / N_KV_HEADS);
    pair_index = int'(active.layer)*N_KV_HEADS + cache_head;
    k_cache = '{rgn: RGN_HBM, off: AXI_AW'(kv_pc_off(active.layer, 4'(cache_head), 1'b0))};
    v_cache = '{rgn: RGN_HBM, off: AXI_AW'(kv_pc_off(active.layer, 4'(cache_head), 1'b1))};
    k_scales = '{rgn: RGN_HBM,
                 off: KV_SCALE_BASE + AXI_AW'(pair_index*KV_SCALE_BYTES)};
    v_scales = offset_addr(k_scales, K_SCALE_BYTES);

    mm.cmd = '0;
    mm.cmd.m = 16'd1;
    mm.cmd_valid = 1'b0;
    case (state)
      PROJECT_QKV: begin
        mm.cmd.a_base = active.x_base;
        mm.cmd.b_base = active.w_base;
        mm.cmd.s_base = active.s_base;
        mm.cmd.c_base = QKV_VEC;
        mm.cmd.k = 16'(D_MODEL);
        mm.cmd.n = 16'(QKV_ELEMENTS);
        mm.cmd.out_fmt = OUT_VEC;
        mm.cmd_valid = 1'b1;
      end
      SCORE_MATMUL: begin
        mm.cmd.a_base = offset_addr(QKV_I8, int'(head)*HEAD_BYTES);
        mm.cmd.b_base = k_cache;
        mm.cmd.s_base = k_scales;
        mm.cmd.c_base = SCORES;
        mm.cmd.k = 16'(HEAD_DIM);
        mm.cmd.n = active.pos + 16'd1;
        mm.cmd.b_is_kv = 1'b1;
        mm.cmd.out_fmt = OUT_VEC;
        mm.cmd_valid = 1'b1;
      end
      VALUE_MATMUL: begin
        mm.cmd.a_base = PROBS;
        mm.cmd.b_base = v_cache;
        mm.cmd.s_base = v_scales;
        mm.cmd.c_base = offset_addr(HEADS, int'(head)*HEAD_BYTES);
        mm.cmd.k = active.pos + 16'd1;
        mm.cmd.n = 16'(HEAD_DIM);
        mm.cmd.b_is_kv = 1'b1;
        mm.cmd.b_row_stride = 32'(V_N_TILE_STRIDE);
        mm.cmd.out_fmt = OUT_I8;
        mm.cmd_valid = 1'b1;
      end
      PROJECT_OUTPUT: begin
        mm.cmd.a_base = HEADS;
        mm.cmd.b_base = offset_addr(active.w_base, WO_OFFSET);
        mm.cmd.s_base = offset_addr(active.s_base, WO_SCALE_OFFSET);
        mm.cmd.c_base = active.out_base;
        mm.cmd.k = 16'(D_MODEL);
        mm.cmd.n = 16'(D_MODEL);
        mm.cmd.out_fmt = OUT_VEC;
        mm.cmd_valid = 1'b1;
      end
      default: ;
    endcase

    port_sel = PORT_IDLE;
    case (state)
      START_ROPE, WAIT_ROPE: port_sel = PORT_ROPE;
      START_SOFTMAX, WAIT_SOFTMAX: port_sel = PORT_SOFTMAX;
      KV_READ_REQ, KV_READ: port_sel = PORT_KV;
      default: ;
    endcase

    // Source consists of one K row and one V row for the current KV head.
    kv_src.req = linear_req(offset_addr(QKV_I8, (N_HEADS + int'(head))*HEAD_BYTES),
                            HEAD_BEATS);
    kv_src.req.n_rows = 16'd2;
    kv_src.req.row_stride = N_KV_HEADS * HEAD_BYTES;
    kv_src.req_valid = (state == KV_READ_REQ);
    kv_src.ready = (state == KV_READ);

    // llm_pkg KV contract: one request per vector; base.off[28:0] carries kv_tag_t and T4
    // scatters the HEAD_BEATS-beat payload into the K^T (column) or V (row) tiles.
    kv_tag = '{layer: active.layer, kvh: 4'(head), is_v: kv_is_v, pos: active.pos};
    kv.req = linear_req('{rgn: RGN_HBM, off: AXI_AW'(kv_tag)}, HEAD_BEATS);
    kv.req_valid = (state == KV_WRITE_REQ);
    kv.data = kv_payload[(int'(kv_is_v)*HEAD_BEATS + int'(kv_beat))*BEAT_W +: BEAT_W];
    kv.valid = (state == KV_WRITE);
    kv.last = kv.valid && (int'(kv_beat) == HEAD_BEATS-1);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      active <= '0;
      head <= '0;
      kv_beat <= '0;
      kv_payload <= '0;
      kv_committed <= 1'b0;
      kv_is_v <= 1'b0;
    end else begin
      case (state)
        IDLE: if (cmd_valid && cmd_ready) begin
          active <= cmd;
          head <= '0;
          state <= PROJECT_QKV;
        end
        PROJECT_QKV: if (mm.cmd_ready) state <= WAIT_QKV;
        WAIT_QKV: if (mm.done) state <= START_ROPE;
        START_ROPE: state <= WAIT_ROPE;
        WAIT_ROPE: if (rope_done) state <= KV_READ_REQ;
        KV_READ_REQ: if (kv_src.req_ready) begin
          kv_beat <= '0;
          state <= KV_READ;
        end
        KV_READ: if (kv_src.valid) begin
          kv_payload[int'(kv_beat)*BEAT_W +: BEAT_W] <= kv_src.data;
          kv_beat <= kv_beat + 1'b1;
          if (kv_src.last) begin
            kv_is_v <= 1'b0;
            state <= KV_WRITE_REQ;
          end
        end
        KV_WRITE_REQ: if (kv.req_ready) begin
          kv_beat <= '0;
          state <= KV_WRITE;
        end
        KV_WRITE: if (kv.ready) begin
          if (kv.last) begin
            kv_committed <= kv.done;
            state <= KV_COMMIT;
          end else kv_beat <= kv_beat + 1'b1;
        end
        KV_COMMIT: if (kv.done || kv_committed) begin
          if (!kv_is_v) begin                    // K committed -> append V for this head
            kv_is_v <= 1'b1;
            kv_committed <= 1'b0;
            state <= KV_WRITE_REQ;
          end else if (int'(head) == N_KV_HEADS-1) begin
            head <= '0;
            state <= SCORE_MATMUL;
          end else begin
            head <= head + 1'b1;
            state <= KV_READ_REQ;
          end
        end
        SCORE_MATMUL: if (mm.cmd_ready) state <= WAIT_SCORES;
        WAIT_SCORES: if (mm.done) state <= START_SOFTMAX;
        START_SOFTMAX: state <= WAIT_SOFTMAX;
        WAIT_SOFTMAX: if (sm_done) state <= VALUE_MATMUL;
        VALUE_MATMUL: if (mm.cmd_ready) state <= WAIT_VALUES;
        WAIT_VALUES: if (mm.done) begin
          if (int'(head) == N_HEADS-1) state <= PROJECT_OUTPUT;
          else begin
            head <= head + 1'b1;
            state <= SCORE_MATMUL;
          end
        end
        PROJECT_OUTPUT: if (mm.cmd_ready) state <= WAIT_OUTPUT;
        WAIT_OUTPUT: if (mm.done) state <= FINISH;
        FINISH: state <= IDLE;
        default: state <= IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk) if (rst_n && cmd_valid && cmd_ready) begin
    assert (int'(cmd.layer) < N_LAYERS && int'(cmd.pos) < MAX_SEQ)
      else $fatal(1, "Invalid attention layer or decode position");
  end
`endif
endmodule

module rope_unit
  import llm_pkg::*;
  import t2_attn_pkg::*;
#(
  parameter addr_t SRC = '0, DST = '0,
  parameter logic [AXI_AW-1:0] ROPE_BASE = '0
)(
  input logic clk, rst_n, start,
  input logic [15:0] pos,
  output logic done,
  mem_rd_if.client rd,
  mem_wr_if.client wr
);
  localparam int PAIRS = HEAD_DIM / 2;
  localparam int HEADS = N_HEADS + 2*N_KV_HEADS;
  localparam int VEC_BEATS = HEAD_VEC_BYTES / BEAT_BYTES;
  localparam int COEFF_BEATS = COEFF_BYTES / BEAT_BYTES;
  localparam int PAIR_W = (PAIRS > 1) ? $clog2(PAIRS) : 1;
  localparam int HEAD_W = (HEADS > 1) ? $clog2(HEADS) : 1;
  localparam int READ_BEAT_W = ($clog2(VEC_BEATS) > 0) ? $clog2(VEC_BEATS) : 1;
  localparam int WRITE_BEAT_W = ($clog2(HEAD_BEATS) > 0) ? $clog2(HEAD_BEATS) : 1;
  typedef enum logic [3:0] {
    IDLE, COEFF_REQ, READ_COEFF, HEAD_REQ, READ_HEAD,
    ROTATE, WRITE_REQ, WRITE_HEAD, COMMIT, FINISH
  } state_e;
  state_e state;
  logic [15:0] position;
  logic [HEAD_W-1:0] head;
  logic [PAIR_W-1:0] pair_idx;
  logic [READ_BEAT_W-1:0] read_beat;
  logic [WRITE_BEAT_W-1:0] write_beat;
  logic [COEFF_BYTES*8-1:0] coeffs;
  logic [HEAD_DIM*VEC_W-1:0] input_head;
  logic [HEAD_DIM*DATA_W-1:0] output_head;
  logic committed;
  logic signed [VEC_W-1:0] x0, x1;
  logic signed [15:0] cos_coeff, sin_coeff;
  logic signed [32:0] y0, y1;

  function automatic logic [DATA_W-1:0] round_i8(input logic signed [32:0] value);
    localparam int SHIFT = QKV_FRAC + COEFF_FRAC;
    logic [32:0] magnitude, rounded;
    magnitude = (value < 0) ? $unsigned(-value) : $unsigned(value);
    rounded = (magnitude + (33'd1 << (SHIFT-1))) >> SHIFT;
    if (value < 0) return (rounded >= 128) ? 8'h80 : DATA_W'(-rounded);
    return (rounded > 127) ? 8'h7f : DATA_W'(rounded);
  endfunction

  assign x0 = $signed(input_head[int'(pair_idx)*VEC_W +: VEC_W]);
  assign x1 = $signed(input_head[(int'(pair_idx)+PAIRS)*VEC_W +: VEC_W]);
  assign cos_coeff = $signed(coeffs[int'(pair_idx)*16 +: 16]);
  assign sin_coeff = $signed(coeffs[(int'(pair_idx)+PAIRS)*16 +: 16]);
  always_comb begin
    if (position == 0 || int'(head) >= N_HEADS+N_KV_HEADS) begin
      y0 = 33'(x0) <<< COEFF_FRAC;
      y1 = 33'(x1) <<< COEFF_FRAC;
    end else begin
      y0 = x0*cos_coeff - x1*sin_coeff;
      y1 = x0*sin_coeff + x1*cos_coeff;
    end
  end

  assign rd.req = (state == COEFF_REQ)
                ? linear_req('{rgn: RGN_HBM, off: ROPE_BASE + AXI_AW'(int'(position)*COEFF_BYTES)},
                             COEFF_BEATS)
                : linear_req(offset_addr(SRC, int'(head)*HEAD_VEC_BYTES), VEC_BEATS);
  assign rd.req_valid = (state == COEFF_REQ || state == HEAD_REQ);
  assign rd.ready = (state == READ_COEFF || state == READ_HEAD);
  assign wr.req = linear_req(offset_addr(DST, int'(head)*HEAD_BYTES), HEAD_BEATS);
  assign wr.req_valid = (state == WRITE_REQ);
  assign wr.data = output_head[int'(write_beat)*BEAT_W +: BEAT_W];
  assign wr.valid = (state == WRITE_HEAD);
  assign wr.last = wr.valid && (int'(write_beat) == HEAD_BEATS-1);
  assign done = (state == FINISH);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      position <= '0;
      head <= '0;
      pair_idx <= '0;
      read_beat <= '0;
      write_beat <= '0;
      coeffs <= '0;
      input_head <= '0;
      output_head <= '0;
      committed <= 1'b0;
    end else begin
      case (state)
        IDLE: if (start) begin
          position <= pos;
          head <= '0;
          read_beat <= '0;
          state <= (pos == 0) ? HEAD_REQ : COEFF_REQ;
        end
        COEFF_REQ: if (rd.req_ready) state <= READ_COEFF;
        READ_COEFF: if (rd.valid) begin
          coeffs[int'(read_beat)*BEAT_W +: BEAT_W] <= rd.data;
          read_beat <= read_beat + 1'b1;
          if (rd.last) begin
            read_beat <= '0;
            state <= HEAD_REQ;
          end
        end
        HEAD_REQ: if (rd.req_ready) state <= READ_HEAD;
        READ_HEAD: if (rd.valid) begin
          input_head[int'(read_beat)*BEAT_W +: BEAT_W] <= rd.data;
          read_beat <= read_beat + 1'b1;
          if (rd.last) begin
            read_beat <= '0;
            pair_idx <= '0;
            state <= ROTATE;
          end
        end
        ROTATE: begin
          output_head[int'(pair_idx)*DATA_W +: DATA_W] <= round_i8(y0);
          output_head[(int'(pair_idx)+PAIRS)*DATA_W +: DATA_W] <= round_i8(y1);
          if (int'(pair_idx) == PAIRS-1) state <= WRITE_REQ;
          else pair_idx <= pair_idx + 1'b1;
        end
        WRITE_REQ: if (wr.req_ready) begin
          write_beat <= '0;
          state <= WRITE_HEAD;
        end
        WRITE_HEAD: if (wr.ready) begin
          if (wr.last) begin
            committed <= wr.done;
            state <= COMMIT;
          end
          else write_beat <= write_beat + 1'b1;
        end
        COMMIT: if (wr.done || committed) begin
          if (int'(head) == HEADS-1) state <= FINISH;
          else begin
            head <= head + 1'b1;
            state <= HEAD_REQ;
          end
        end
        FINISH: state <= IDLE;
        default: state <= IDLE;
      endcase
    end
  end
endmodule

module softmax_unit
  import llm_pkg::*;
  import t2_attn_pkg::*;
#(
  parameter addr_t SRC = '0, DST = '0
)(
  input logic clk, rst_n, start,
  input logic [15:0] len,
  output logic done,
  mem_rd_if.client rd,
  mem_wr_if.client wr
);
  localparam int IN_LANES = BEAT_W / VEC_W;
  localparam int OUT_LANES = BEAT_W / DATA_W;
  localparam int INDEX_W = (MAX_SEQ > 1) ? $clog2(MAX_SEQ) : 1;
  localparam int EXP_FRAC = 16;
  localparam int EXP_W = EXP_FRAC + 1;
  localparam int SUM_W = $clog2(MAX_SEQ+1) + EXP_FRAC;
  localparam int DIV_W = (SUM_W > EXP_W+7 ? SUM_W : EXP_W+7) + 7;
  localparam int LUT_STEP_FRAC = SCORE_FRAC - 2;
  localparam int INTERP_W = EXP_W + LUT_STEP_FRAC;
  typedef enum logic [3:0] {
    IDLE, READ_REQ, READ_BEAT, SCAN, WRITE_REQ,
    PROB_READ, PROB_INIT, DIVIDE, WRITE_BEAT, COMMIT, FINISH
  } state_e;
  state_e state;
  logic sum_pass;
  logic [15:0] length;
  logic [INDEX_W-1:0] index;
  logic last_element;
  logic [BEAT_W-1:0] score_beat, prob_beat;
  logic signed [VEC_W-1:0] max_score, score;
  logic [VEC_W:0] delta;
  logic [EXP_W-1:0] exp_value, selected_exp;
  (* ram_style = "block" *) logic [EXP_W-1:0] exp_mem [MAX_SEQ];
  logic [SUM_W-1:0] exp_sum;
  logic [DIV_W-1:0] remainder, trial;
  logic [2:0] bit_idx;
  logic [6:0] quotient, next_quotient;

  // exp(-i/4) in Q16, rounded to nearest. Interpolate inside each interval.
  localparam logic [EXP_W-1:0] EXP_LUT [0:64] = '{
    17'd65536, 17'd51039, 17'd39750, 17'd30957, 17'd24109, 17'd18776, 17'd14623, 17'd11388,
    17'd8869, 17'd6907, 17'd5380, 17'd4190, 17'd3263, 17'd2541, 17'd1979, 17'd1541,
    17'd1200, 17'd935, 17'd728, 17'd567, 17'd442, 17'd344, 17'd268, 17'd209,
    17'd162, 17'd127, 17'd99, 17'd77, 17'd60, 17'd47, 17'd36, 17'd28,
    17'd22, 17'd17, 17'd13, 17'd10, 17'd8, 17'd6, 17'd5, 17'd4,
    17'd3, 17'd2, 17'd2, 17'd1, 17'd1, 17'd1, 17'd1, 17'd1,
    17'd0, 17'd0, 17'd0, 17'd0, 17'd0, 17'd0, 17'd0, 17'd0,
    17'd0, 17'd0, 17'd0, 17'd0, 17'd0, 17'd0, 17'd0, 17'd0,
    17'd0
  };

  function automatic logic [EXP_W-1:0] exp_neg(input logic [VEC_W:0] x);
    logic [6:0] cell_idx;
    logic [LUT_STEP_FRAC-1:0] fraction;
    logic [EXP_W-1:0] hi, lo;
    logic [EXP_W-1:0] span;
    logic [INTERP_W-1:0] correction;
    if (x >= (17'd16 << SCORE_FRAC)) return '0;
    cell_idx = 7'(x >> LUT_STEP_FRAC);
    fraction = x[LUT_STEP_FRAC-1:0];
    hi = EXP_LUT[cell_idx];
    lo = EXP_LUT[cell_idx + 7'd1];
    span = hi-lo;
    correction = span*fraction + (INTERP_W'(1) << (LUT_STEP_FRAC-1));
    return hi - EXP_W'(correction >> LUT_STEP_FRAC);
  endfunction

  assign score = $signed(score_beat[(int'(index) % IN_LANES)*VEC_W +: VEC_W]);
  assign delta = $unsigned((VEC_W+1)'(max_score) - (VEC_W+1)'(score));
  assign exp_value = exp_neg(delta);
  assign trial = DIV_W'(exp_sum) << bit_idx;
  assign last_element = (16'(index) == length-16'd1);
  always_comb begin
    next_quotient = quotient;
    next_quotient[bit_idx] = (remainder >= trial);
  end

  assign rd.req = linear_req(SRC, (int'(length) + IN_LANES-1) / IN_LANES);
  assign rd.req_valid = (state == READ_REQ);
  assign rd.ready = (state == READ_BEAT);
  assign wr.req = linear_req(DST, (int'(length) + OUT_LANES-1) / OUT_LANES);
  assign wr.req_valid = (state == WRITE_REQ);
  assign wr.data = prob_beat;
  assign wr.valid = (state == WRITE_BEAT);
  assign wr.last = wr.valid && last_element;
  assign done = (state == FINISH);

  // One synchronous read port. Only initialized entries (index < len) are read.
  always_ff @(posedge clk) begin
    if (state == SCAN && sum_pass) exp_mem[index] <= exp_value;
    if (state == PROB_READ) selected_exp <= exp_mem[index];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      sum_pass <= 1'b0;
      length <= '0;
      index <= '0;
      score_beat <= '0;
      prob_beat <= '0;
      max_score <= '0;
      exp_sum <= '0;
      remainder <= '0;
      bit_idx <= '0;
      quotient <= '0;
    end else begin
      case (state)
        IDLE: if (start) begin
          length <= len;
          index <= '0;
          sum_pass <= 1'b0;
          max_score <= {1'b1, {(VEC_W-1){1'b0}}};
          exp_sum <= '0;
          prob_beat <= '0;
          state <= READ_REQ;
        end
        READ_REQ: if (rd.req_ready) state <= READ_BEAT;
        READ_BEAT: if (rd.valid) begin
          score_beat <= rd.data;
          state <= SCAN;
        end
        SCAN: begin
          if (sum_pass) exp_sum <= exp_sum + SUM_W'(exp_value);
          else if (score > max_score) max_score <= score;
          if (last_element) begin
            index <= '0;
            state <= sum_pass ? WRITE_REQ : READ_REQ;
            sum_pass <= 1'b1;
          end else begin
            index <= index + 1'b1;
            if (int'(index) % IN_LANES == IN_LANES-1) state <= READ_BEAT;
          end
        end
        WRITE_REQ: if (wr.req_ready) state <= PROB_READ;
        PROB_READ: state <= PROB_INIT;
        PROB_INIT: begin
          // Round 127*exp/sum with one seven-cycle restoring divider.
          remainder <= (DIV_W'(selected_exp) << 7) - DIV_W'(selected_exp)
                       + (DIV_W'(exp_sum) >> 1);
          bit_idx <= 3'd6;
          quotient <= '0;
          state <= DIVIDE;
        end
        DIVIDE: begin
          if (remainder >= trial) remainder <= remainder - trial;
          quotient <= next_quotient;
          if (bit_idx == 0) begin
            prob_beat[(int'(index) % OUT_LANES)*DATA_W +: DATA_W] <= {1'b0, next_quotient};
            if (last_element || int'(index) % OUT_LANES == OUT_LANES-1) state <= WRITE_BEAT;
            else begin
              index <= index + 1'b1;
              state <= PROB_READ;
            end
          end else bit_idx <= bit_idx - 1'b1;
        end
        WRITE_BEAT: if (wr.ready) begin
          if (last_element) state <= wr.done ? FINISH : COMMIT;
          else begin
            index <= index + 1'b1;
            prob_beat <= '0;
            state <= PROB_READ;
          end
        end
        COMMIT: if (wr.done) state <= FINISH;
        FINISH: state <= IDLE;
        default: state <= IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk) if (rst_n && state == IDLE && start) begin
    assert (len > 0 && int'(len) <= MAX_SEQ)
      else $fatal(1, "Invalid softmax length");
  end
`endif
endmodule
