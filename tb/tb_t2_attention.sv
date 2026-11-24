`timescale 1ns/1ps
module tb_t2_attention;
  import llm_pkg::*;
  import t2_attn_pkg::*;

  localparam logic [AXI_AW-1:0] KV_SCALE_BASE = AXI_AW'('h5000000);
  localparam logic [AXI_AW-1:0] ROPE_BASE = AXI_AW'('h6000000);
  localparam int QKV_I8_OFF = QKV_VEC_BYTES;
  localparam int SCORES_OFF = QKV_I8_OFF + QKV_BYTES;
  localparam int PROBS_OFF = SCORES_OFF + SCORE_BYTES;
  localparam int HEADS_OFF = PROBS_OFF + PROB_BYTES;
  localparam int PAIR_SCALE_BYTES = (MAX_SEQ+HEAD_DIM)*4;
  localparam addr_t X_BASE = '{rgn: RGN_ACT, off: 'h100};
  localparam addr_t W_BASE = '{rgn: RGN_HBM, off: 'h500000};
  localparam addr_t S_BASE = '{rgn: RGN_HBM, off: 'h1000000};
  localparam addr_t OUT_BASE = '{rgn: RGN_ACT, off: 'h2000};

  logic clk = 0, rst_n = 0;
  always #5 clk = !clk;
  attn_cmd_t cmd;
  logic cmd_valid, cmd_ready, done;
  mm_if mm();
  mem_rd_if rd();
  mem_wr_if wr(), kv();
  attn_unit #(.KV_SCALE_BASE(KV_SCALE_BASE), .ROPE_BASE(ROPE_BASE)) dut (
    .clk, .rst_n, .cmd, .cmd_valid, .cmd_ready, .done, .mm, .rd, .wr, .kv
  );

  logic [BEAT_W-1:0] scratch [SCRATCH_BYTES / BEAT_BYTES];
  logic signed [15:0] cosines [HEAD_DIM/2], sines [HEAD_DIM/2];
  mem_req_t read_req, write_req;
  int cycle, read_beat, write_beat, kv_beat, mm_delay, wr_delay, kv_delay;
  int pos, layer, mm_count, coeff_reads;
  logic [31:0] kv_count;
  logic reading, writing, kv_writing;
  logic [31:0] rng;
  int read_total, read_index;
  int expected_kv_off, test_qh;
  logic [BEAT_W-1:0] expected_value;

  assign read_total = int'(read_req.n_rows)*int'(read_req.row_beats);
  assign read_index = !reading ? 0 : int'(read_req.base.off)/BEAT_BYTES
                      + (read_beat / int'(read_req.row_beats))*int'(read_req.row_stride)/BEAT_BYTES
                      + (read_beat % int'(read_req.row_beats));
  assign rd.req_ready = !reading && rng[0];
  assign rd.last = rd.valid && read_beat == read_total-1;
  always_comb begin
    rd.data = '0;
    if (reading) begin
      if (read_req.base.rgn == RGN_SCRATCH) rd.data = scratch[read_index];
      else begin
        for (int lane = 0; lane < BEAT_W/16; lane++)
          rd.data[lane*16 +: 16] = coefficient(read_beat*(BEAT_W/16)+lane);
      end
    end
  end

  assign wr.req_ready = !writing && wr_delay == 0 && rng[1];
  assign wr.ready = writing && rng[2] && (cycle % 13 < 8);
  assign kv.req_ready = !kv_writing && kv_delay == 0 && rng[3];
  assign kv.ready = kv_writing && rng[4] && (cycle % 17 < 11);
  assign mm.cmd_ready = mm_delay == 0 && rng[5];

  function automatic logic signed [15:0] coefficient(input int index);
    return index < HEAD_DIM/2 ? cosines[index] : sines[index-HEAD_DIM/2];
  endfunction

  function automatic int projected_value(input int head, lane);
    if (lane == 0) return 32767;
    if (lane == HEAD_DIM/2) return -32768;
    return ((head*7 + lane*3)%29 - 14)*128 + (lane%2 == 0 ? 64 : -64);
  endfunction

  function automatic logic [7:0] rope_oracle(input int head, lane);
    int pair_idx, x, y, rounded;
    real rotated;
    pair_idx = lane % (HEAD_DIM/2);
    x = projected_value(head, pair_idx);
    y = projected_value(head, pair_idx+HEAD_DIM/2);
    if (pos == 0 || head >= N_HEADS+N_KV_HEADS)
      rotated = real'(projected_value(head, lane)) / 128.0;
    else if (lane < HEAD_DIM/2)
      rotated = (real'(x)*real'(cosines[pair_idx]) - real'(y)*real'(sines[pair_idx])) / 4194304.0;
    else
      rotated = (real'(x)*real'(sines[pair_idx]) + real'(y)*real'(cosines[pair_idx])) / 4194304.0;
    rounded = $rtoi(rotated < 0 ? rotated-0.5 : rotated+0.5);
    if (rounded < -128) rounded = -128;
    if (rounded > 127) rounded = 127;
    return 8'(rounded);
  endfunction

  function automatic int score_code(input int head, token);
    case (head%4)
      0: return -1024;
      1: return token == pos ? 0 : -2048;
      2: return token == 0 ? -32768 : token == 1 ? 32767 : 0;
      default: return -(token%17)*37 - 16;
    endcase
  endfunction

  function automatic logic [7:0] head_value(input int head, lane);
    return 8'(((head+1)*7 + lane)%127 - 63);
  endfunction

  function automatic int exp_oracle(input int delta);
    int interval, upper, lower;
    if (delta >= 4096) return 0;
    interval = delta / 64;
    upper = $rtoi($exp(-real'(interval)/4.0)*65536.0+0.5);
    lower = $rtoi($exp(-real'(interval+1)/4.0)*65536.0+0.5);
    return upper - ((upper-lower)*(delta%64)+32)/64;
  endfunction

  task automatic check_matrix_command;
    int head, cache_head, pair_idx, v_idx, high, actual, expected, ideal, exp_total, delta;
    real sum;
    addr_t k_addr, v_addr, ks_addr, vs_addr;
    mm_cmd_t want;
    want = '0;
    want.m = 16'd1;
    if (mm_count == 0) begin
      want.a_base = X_BASE;
      want.b_base = W_BASE;
      want.s_base = S_BASE;
      want.c_base = '{rgn: RGN_SCRATCH, off: 0};
      want.k = 16'(D_MODEL);
      want.n = 16'(QKV_ELEMENTS);
      want.out_fmt = OUT_VEC;
      for (int h = 0; h < N_HEADS+2*N_KV_HEADS; h++)
        for (int i = 0; i < HEAD_DIM; i++) begin
          v_idx = h*HEAD_DIM+i;
          scratch[v_idx/(BEAT_W/VEC_W)][(v_idx%(BEAT_W/VEC_W))*VEC_W +: VEC_W]
            = VEC_W'(projected_value(h, i));
        end
    end else if (mm_count == 2*N_HEADS+1) begin
      want.a_base = '{rgn: RGN_SCRATCH, off: AXI_AW'(HEADS_OFF)};
      want.b_base = offset_addr(W_BASE, D_MODEL*QKV_ELEMENTS/N_HBM);   // per-PC bytes
      want.s_base = offset_addr(S_BASE, QKV_ELEMENTS*4);
      want.c_base = OUT_BASE;
      want.k = 16'(D_MODEL);
      want.n = 16'(D_MODEL);
      want.out_fmt = OUT_VEC;
      for (int h = 0; h < N_HEADS; h++)
        for (int i = 0; i < HEAD_DIM; i++) begin
          v_idx = HEADS_OFF + h*HEAD_BYTES+i;
          if (scratch[v_idx/BEAT_BYTES][(v_idx%BEAT_BYTES)*8 +: 8] !== head_value(h, i))
            $fatal(1, "Incorrect head concatenation at head %0d lane %0d", h, i);
        end
    end else begin
      if (kv_count != 2*N_KV_HEADS) $fatal(1, "Score or value GEMM before KV commits");
      head = (mm_count-1) / 2;
      cache_head = head / (N_HEADS/N_KV_HEADS);
      pair_idx = layer*N_KV_HEADS + cache_head;
      k_addr = '{rgn:RGN_HBM, off:AXI_AW'(kv_pc_off(8'(layer), 4'(cache_head), 1'b0))};
      v_addr = '{rgn:RGN_HBM, off:AXI_AW'(kv_pc_off(8'(layer), 4'(cache_head), 1'b1))};
      ks_addr = '{rgn:RGN_HBM, off:KV_SCALE_BASE + AXI_AW'(pair_idx*PAIR_SCALE_BYTES)};
      vs_addr = offset_addr(ks_addr, MAX_SEQ*4);
      want.b_is_kv = 1'b1;
      if (mm_count%2 == 1) begin
        want.a_base = '{rgn:RGN_SCRATCH, off:AXI_AW'(QKV_I8_OFF+head*HEAD_BYTES)};
        want.b_base = k_addr;
        want.s_base = ks_addr;
        want.c_base = '{rgn:RGN_SCRATCH, off:AXI_AW'(SCORES_OFF)};
        want.k = 16'(HEAD_DIM);
        want.n = 16'(pos+1);
        want.out_fmt = OUT_VEC;
        for (int i = 0; i < HEAD_DIM; i++) begin
          v_idx = QKV_I8_OFF + head*HEAD_BYTES+i;
          if (scratch[v_idx/BEAT_BYTES][(v_idx%BEAT_BYTES)*8 +: 8] !== rope_oracle(head, i))
            $fatal(1, "Q RoPE mismatch at position %0d head %0d lane %0d", pos, head, i);
        end
        for (int i = 0; i <= pos; i++) begin
          v_idx = SCORES_OFF + i*2;
          scratch[v_idx/BEAT_BYTES][(v_idx%BEAT_BYTES)*8 +: 16] = VEC_W'(score_code(head, i));
        end
      end else begin
        want.a_base = '{rgn:RGN_SCRATCH, off:AXI_AW'(PROBS_OFF)};
        want.b_base = v_addr;
        want.s_base = vs_addr;
        want.c_base = '{rgn:RGN_SCRATCH, off:AXI_AW'(HEADS_OFF+head*HEAD_BYTES)};
        want.k = 16'(pos+1);
        want.n = 16'(HEAD_DIM);
        want.b_row_stride = (MAX_SEQ/ARR_R)*(TILE_BYTES/N_HBM);   // per-PC
        want.out_fmt = OUT_I8;
        high = -32768;
        for (int i = 0; i <= pos; i++) if (score_code(head, i) > high) high = score_code(head, i);
        sum = 0.0;
        exp_total = 0;
        for (int i = 0; i <= pos; i++) begin
          delta = high-score_code(head, i);
          sum += $exp(-real'(delta)/256.0);
          exp_total += exp_oracle(delta);
        end
        for (int i = 0; i < ((pos+BEAT_BYTES)/BEAT_BYTES)*BEAT_BYTES; i++) begin
          v_idx = PROBS_OFF+i;
          actual = int'(scratch[v_idx/BEAT_BYTES][(v_idx%BEAT_BYTES)*8 +: 8]);
          delta = high-score_code(head, i);
          expected = i <= pos ? (127*exp_oracle(delta)+exp_total/2)/exp_total : 0;
          ideal = i <= pos ? $rtoi(127.0*$exp(-real'(delta)/256.0)/sum+0.5) : 0;
          if (actual != expected)
            $fatal(1, "Softmax pos %0d head %0d lane %0d got %0d expected %0d",
                   pos, head, i, actual, expected);
          if (actual < ideal-1 || actual > ideal+1)
            $fatal(1, "Softmax pos %0d head %0d lane %0d got %0d floating point rounded to %0d",
                   pos, head, i, actual, ideal);
        end
        for (int i = 0; i < HEAD_DIM; i++) begin
          v_idx = HEADS_OFF+head*HEAD_BYTES+i;
          scratch[v_idx/BEAT_BYTES][(v_idx%BEAT_BYTES)*8 +: 8] = head_value(head, i);
        end
      end
    end
    if (mm.cmd !== want)
      $fatal(1, "Unexpected GEMM %0d got %h expected %h", mm_count, mm.cmd, want);
  endtask

  always @(posedge clk) begin
    if (!rst_n) begin
      cycle <= 0;
      rng <= 32'h5a8176fe;
      reading <= 0;
      writing <= 0;
      kv_writing <= 0;
      read_req <= '0;
      write_req <= '0;
      read_beat <= 0;
      write_beat <= 0;
      kv_beat <= 0;
      mm_delay <= 0;
      wr_delay <= 0;
      kv_delay <= 0;
      mm_count <= 0;
      kv_count <= 0;
      coeff_reads <= 0;
      rd.valid <= 0;
      wr.done <= 0;
      kv.done <= 0;
      mm.done <= 0;
      mm.done_tag <= '0;
    end else begin
      cycle <= cycle+1;
      rng <= {rng[30:0], rng[31]^rng[21]^rng[1]^rng[0]};
      wr.done <= wr_delay == 1;
      kv.done <= kv_delay == 1;
      mm.done <= mm_delay == 1;
      if (wr_delay > 0) wr_delay <= wr_delay-1;
      if (kv_delay > 0) kv_delay <= kv_delay-1;
      if (mm_delay > 0) mm_delay <= mm_delay-1;
      if (cmd_valid && cmd_ready) begin
        mm_count <= 0;
        kv_count <= 0;
        coeff_reads <= 0;
      end
      if (rd.req_valid && rd.req_ready) begin
        if (rd.req.base.rgn == RGN_HBM) begin
          if (pos == 0 || rd.req !== linear_req(
              '{rgn:RGN_HBM, off:ROPE_BASE + AXI_AW'(pos*COEFF_BYTES)}, COEFF_BYTES/BEAT_BYTES))
            $fatal(1, "Unexpected RoPE coefficient read");
          coeff_reads <= coeff_reads+1;
        end else if (rd.req.base.rgn != RGN_SCRATCH)
          $fatal(1, "Unexpected read region");
        read_req <= rd.req;
        read_beat <= 0;
        reading <= 1;
      end
      if (rd.valid && rd.ready) begin
        if (rd.last) reading <= 0;
        else read_beat <= read_beat+1;
        rd.valid <= 0;
      end else if (reading && !rd.valid && rng[6]) rd.valid <= 1;
      if (wr.req_valid && wr.req_ready) begin
        if (wr.req.base.rgn != RGN_SCRATCH || wr.req.n_rows != 1)
          $fatal(1, "Unexpected scratch write request");
        write_req <= wr.req;
        write_beat <= 0;
        writing <= 1;
      end
      if (wr.valid && wr.ready) begin
        scratch[int'(write_req.base.off)/BEAT_BYTES+write_beat] = wr.data;
        if (wr.last != (write_beat == int'(write_req.row_beats)-1)) $fatal(1, "Wrong wr.last");
        if (wr.last) begin
          writing <= 0;
          wr_delay <= 3;
        end else write_beat <= write_beat+1;
      end
      if (kv.req_valid && kv.req_ready) begin
        // llm_pkg KV contract: one request per vector, base.off = kv_tag_t
        expected_kv_off = int'(kv_tag_t'{layer: 8'(layer), kvh: 4'(kv_count/2), is_v: kv_count[0],
                                         pos: 16'(pos)});
        if (kv.req !== linear_req('{rgn:RGN_HBM, off:AXI_AW'(expected_kv_off)}, HEAD_BEATS))
          $fatal(1, "Wrong KV tag for request %0d", kv_count);
        kv_writing <= 1;
        kv_beat <= 0;
      end
      if (kv.valid && kv.ready) begin
        test_qh = N_HEADS + kv_count/2 + (kv_count[0] ? N_KV_HEADS : 0);
        for (int i = 0; i < BEAT_BYTES; i++)
          expected_value[i*8 +: 8] = rope_oracle(test_qh, kv_beat*BEAT_BYTES+i);
        if (kv.data !== expected_value) $fatal(1, "Wrong KV payload head %0d beat %0d", kv_count, kv_beat);
        if (kv.last != (kv_beat == HEAD_BEATS-1)) $fatal(1, "Wrong kv.last");
        if (kv.last) begin
          kv_writing <= 0;
          kv_delay <= 5;
        end else kv_beat <= kv_beat+1;
      end
      if (kv.done) kv_count <= kv_count+1;
      if (mm.cmd_valid && mm.cmd_ready) begin
        check_matrix_command();
        mm_count <= mm_count+1;
        mm_delay <= 5;
      end
    end
  end

  assert property (@(posedge clk) disable iff (!rst_n)
    mm.cmd_valid && !mm.cmd_ready |=> mm.cmd_valid && $stable(mm.cmd));
  assert property (@(posedge clk) disable iff (!rst_n)
    rd.req_valid && !rd.req_ready |=> rd.req_valid && $stable(rd.req));
  assert property (@(posedge clk) disable iff (!rst_n)
    wr.req_valid && !wr.req_ready |=> wr.req_valid && $stable(wr.req));
  assert property (@(posedge clk) disable iff (!rst_n)
    wr.valid && !wr.ready |=> wr.valid && $stable({wr.data, wr.last}));
  assert property (@(posedge clk) disable iff (!rst_n)
    kv.req_valid && !kv.req_ready |=> kv.req_valid && $stable(kv.req));
  assert property (@(posedge clk) disable iff (!rst_n)
    kv.valid && !kv.ready |=> kv.valid && $stable({kv.data, kv.last}));

  task automatic decode(input int token_position, layer_index);
    @(negedge clk);
    pos = token_position;
    layer = layer_index;
    cmd = '{layer:8'(layer), pos:16'(pos), x_base:X_BASE,
            w_base:W_BASE, s_base:S_BASE, out_base:OUT_BASE};
    cmd_valid = 1;
    do @(posedge clk); while (!cmd_ready);
    @(negedge clk);
    cmd_valid = 0;
    cmd = '0;
    do @(negedge clk); while (!done);
    if (mm_count != 2*N_HEADS+2 || kv_count != 2*N_KV_HEADS || mm_delay != 0
        || coeff_reads != (pos == 0 ? 0 : 1))
      $fatal(1, "Wrong transaction counts at completion");
    $display("PASS T2 layer %0d position %0d: %0d GEMMs and %0d KV commits",
             layer, pos, mm_count, kv_count);
  endtask

  initial begin
    for (int i = 0; i < HEAD_DIM/2; i++) begin
      case (i%4)
        0: begin cosines[i] = 0; sines[i] = 32767; end
        1: begin cosines[i] = 0; sines[i] = -32768; end
        2: begin cosines[i] = -32768; sines[i] = 0; end
        default: begin cosines[i] = 23170; sines[i] = 23170; end
      endcase
    end
    cmd = '0;
    cmd_valid = 0;
    repeat (4) @(negedge clk);
    rst_n = 1;
    decode(0, 0);
    decode(34, 2);
    decode(MAX_SEQ-1, N_LAYERS-1);
    $finish;
  end
  initial begin
    #15_000_000;
    $fatal(1, "T2 timeout, pos=%0d GEMMs=%0d KV commits=%0d", pos, mm_count, kv_count);
  end
endmodule
