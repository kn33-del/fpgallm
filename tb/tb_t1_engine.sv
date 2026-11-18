`timescale 1ns/1ps
module t1_read_model import llm_pkg::*; #(
  parameter int W = BEAT_W,
  parameter int BASE = 0,
  parameter region_e REGION = RGN_ACT,
  parameter logic [31:0] SEED = 32'h75134abc,
  parameter int CAPACITY = 4096
)(
  input logic clk, rst_n, stalls,
  mem_rd_if.server bus
);
  logic [W-1:0] mem [CAPACITY];
  mem_req_t req;
  logic active;
  logic [31:0] rng;
  int row, beat, requests, beats;
  assign bus.req_ready = !active && (!stalls || rng[2:0] != 0);
  always @(posedge clk) begin
    if (!rst_n) begin
      active <= 0;
      bus.valid <= 0;
      bus.last <= 0;
      bus.data <= '0;
      req <= '0;
      rng <= SEED;
      row <= 0;
      beat <= 0;
      requests <= 0;
      beats <= 0;
    end else begin
      rng <= {rng[30:0], rng[31]^rng[21]^rng[1]^rng[0]};
      if (bus.req_valid && bus.req_ready) begin
        if (bus.req.base.rgn != REGION || bus.req.row_beats == 0 || bus.req.n_rows == 0)
          $fatal(1, "Bad read request base=%h", bus.req.base);
        req <= bus.req;
        active <= 1;
        row <= 0;
        beat <= 0;
        requests <= requests + 1;
      end
      if (bus.valid && bus.ready) beats <= beats + 1;
      if (bus.valid && bus.ready && bus.last) begin
        active <= 0;
        bus.valid <= 0;
      end else if (active && (!bus.valid || bus.ready)) begin
        bus.valid <= 0;
        if (!stalls || rng[5:3] > 1) begin
          int index;
          index = (int'(req.base.off)-BASE + row*int'(req.row_stride))/(W/8)+beat;
          if (index < 0 || index >= CAPACITY) $fatal(1, "Read outside model memory: %0d", index);
          bus.data <= mem[index];
          bus.valid <= 1;
          bus.last <= (row == int'(req.n_rows)-1 && beat == int'(req.row_beats)-1);
          if (beat == int'(req.row_beats)-1) begin row <= row+1; beat <= 0; end
          else beat <= beat+1;
        end
      end
    end
  end
  assert property (@(posedge clk) disable iff (!rst_n)
    bus.req_valid && !bus.req_ready |=> bus.req_valid && $stable(bus.req));
  assert property (@(posedge clk) disable iff (!rst_n)
    bus.valid && !bus.ready |=> bus.valid && $stable({bus.data,bus.last}));
endmodule

module tb_t1_engine import llm_pkg::*; #(
  parameter int R = 4, C = 32, SHIFT = 16
);
  localparam int A_BASE = 'h1000, B_BASE = 'h10000, S_BASE = 'h8000, C_BASE = 'h20000;
  logic clk = 0;
  always #5 clk = !clk;
  logic rst_n = 0, stalls = 0;
  mm_if cmd();
  mem_rd_if a_rd(), s_rd();
  mem_rd_if #(.W(R*C*DATA_W)) b_rd();
  mem_wr_if c_wr();
  mm_engine #(.R(R), .C(C), .M_MAX(16), .REQUANT_SHIFT(SHIFT)) dut (
    .clk, .rst_n, .cmd, .a_rd, .s_rd, .b_rd, .c_wr
  );
  t1_read_model #(.BASE(A_BASE), .REGION(RGN_ACT), .SEED(32'h13579ace)) a_model (
    .clk, .rst_n, .stalls, .bus(a_rd)
  );
  t1_read_model #(.BASE(S_BASE), .REGION(RGN_HBM), .SEED(32'h2468ace1)) s_model (
    .clk, .rst_n, .stalls, .bus(s_rd)
  );
  t1_read_model #(.W(R*C*DATA_W), .BASE(B_BASE), .REGION(RGN_HBM),
                  .SEED(32'hdeadbeef), .CAPACITY(1024)) b_model (
    .clk, .rst_n, .stalls, .bus(b_rd)
  );
  logic [BEAT_W-1:0] expected [4096];
  logic written [4096];
  logic w_active;
  mem_req_t w_req;
  int w_row, w_beat, done_delay, writes, committed, expected_tiles;
  int case_id, a_count, b_count, s_count, c_count;
  logic [31:0] rng = 32'h932417ae;
  logic throughput_started;
  int throughput_issues;
  assign c_wr.req_ready = !w_active && done_delay == 0 && (!stalls || rng[1:0] != 0);
  // Long blocked bursts exercise backpressure all the way into the array.
  int cycle;
  assign c_wr.ready = w_active && (!stalls || (cycle % 103 < 53 && rng[4:2] != 0));
  always @(posedge clk) begin
    if (!rst_n) begin
      c_wr.done <= 0;
      w_active <= 0;
      w_req <= '0;
      w_row <= 0;
      w_beat <= 0;
      done_delay <= 0;
      writes <= 0;
      committed <= 0;
      cycle <= 0;
      throughput_started <= 0;
      throughput_issues <= 0;
    end else begin
      cycle <= cycle+1;
      rng <= {rng[30:0], rng[31]^rng[21]^rng[1]^rng[0]};
      c_wr.done <= 0;
      if (done_delay > 0) begin
        done_delay <= done_delay-1;
        if (done_delay == 1) begin c_wr.done <= 1; committed <= committed+1; end
      end
      if (c_wr.req_valid && c_wr.req_ready) begin
        if (c_wr.req.base.rgn != RGN_ACT) $fatal(1, "Bad C region");
        w_req <= c_wr.req;
        w_active <= 1;
        w_row <= 0;
        w_beat <= 0;
      end
      if (c_wr.valid && c_wr.ready) begin
        int index;
        logic is_final;
        index = (int'(w_req.base.off)-C_BASE+w_row*int'(w_req.row_stride))/(BEAT_W/8)+w_beat;
        if (index < 0 || index >= c_count) $fatal(1, "C address out of range %0d", index);
        if (written[index]) $fatal(1, "C written twice at %0d", index);
        if (c_wr.data !== expected[index])
          $fatal(1, "case %0d C[%0d]: got %h expected %h", case_id, index, c_wr.data, expected[index]);
        written[index] = 1;
        writes <= writes+1;
        is_final = w_row == int'(w_req.n_rows)-1 && w_beat == int'(w_req.row_beats)-1;
        if (c_wr.last != is_final) $fatal(1, "Wrong C last");
        if (is_final) begin
          w_active <= 0;
          done_delay <= stalls ? 3+int'(rng[3:0]) : 2;
        end else if (w_beat == int'(w_req.row_beats)-1) begin
          w_beat <= 0;
          w_row <= w_row+1;
        end else w_beat <= w_beat+1;
      end
      if (case_id == 0 && dut.arr_a_valid) begin
        throughput_started <= 1;
        throughput_issues <= throughput_issues+1;
      end
      if (case_id == 0 && throughput_started && throughput_issues < 24 && !dut.arr_a_valid)
        $fatal(1, "Decode has a bubble between consecutive K tiles");
    end
  end
  assert property (@(posedge clk) disable iff (!rst_n)
    c_wr.req_valid && !c_wr.req_ready |=> c_wr.req_valid && $stable(c_wr.req));
  assert property (@(posedge clk) disable iff (!rst_n)
    c_wr.valid && !c_wr.ready |=> c_wr.valid && $stable({c_wr.data,c_wr.last}));
  assert property (@(posedge clk) disable iff (!rst_n)
    cmd.cmd_valid && !cmd.cmd_ready |=> cmd.cmd_valid && $stable(cmd.cmd));

  initial begin : test
    string vectors;
    int fd, count, scan, m, k, n, fmt, tag, kv;
    int writes_before, committed_before, a_before, b_before, s_before, expected_issues;
    cmd.cmd_valid = 0;
    cmd.cmd = '0;
    case_id = 0;
    c_count = 0;
    expected_tiles = 0;
    if (!$value$plusargs("vectors=%s", vectors)) $fatal(1, "Missing +vectors=file");
    fd = $fopen(vectors, "r");
    if (fd == 0) $fatal(1, "Cannot read vectors");
    scan = $fscanf(fd, "%d", count);
    repeat (4) @(negedge clk);
    rst_n = 1;
    for (case_id = 0; case_id < count; case_id++) begin
      @(negedge clk);
      scan = $fscanf(fd, "%d %d %d %d %d %d %d %d %d %d",
                     m, k, n, fmt, tag, kv, a_count, b_count, s_count, c_count);
      if (scan != 10) $fatal(1, "Bad vector header");
      for (int i = 0; i < a_count; i++) scan = $fscanf(fd, "%h", a_model.mem[i]);
      for (int i = 0; i < b_count; i++) scan = $fscanf(fd, "%h", b_model.mem[i]);
      for (int i = 0; i < s_count; i++) scan = $fscanf(fd, "%h", s_model.mem[i]);
      for (int i = 0; i < c_count; i++) begin
        scan = $fscanf(fd, "%h", expected[i]);
        written[i] = 0;
      end
      stalls = case_id != 0;
      cmd.cmd.a_base = '{rgn:RGN_ACT, off:AXI_AW'(A_BASE)};
      cmd.cmd.b_base = '{rgn:RGN_HBM, off:AXI_AW'(B_BASE)};
      cmd.cmd.s_base = '{rgn:RGN_HBM, off:AXI_AW'(S_BASE)};
      cmd.cmd.c_base = '{rgn:RGN_ACT, off:AXI_AW'(C_BASE)};
      cmd.cmd.m = 16'(m);
      cmd.cmd.k = 16'(k);
      cmd.cmd.n = 16'(n);
      cmd.cmd.out_fmt = mm_out_e'(fmt);
      cmd.cmd.tag = 4'(tag);
      cmd.cmd.b_is_kv = 1'(kv);
      writes_before = writes;
      committed_before = committed;
      a_before = a_model.beats;
      b_before = b_model.beats;
      s_before = s_model.beats;
      expected_tiles = (n+C-1)/C;
      cmd.cmd_valid = 1;
      do @(posedge clk); while (!cmd.cmd_ready);
      @(negedge clk);
      cmd.cmd_valid = 0;
      do @(negedge clk); while (!cmd.done);
      if (cmd.done_tag != 4'(tag)) $fatal(1, "Wrong done tag");
      if (w_active || done_delay != 0 || committed-committed_before != expected_tiles)
        $fatal(1, "done before writes committed");
      if (writes-writes_before != c_count) $fatal(1, "Missing C writes");
      if (a_model.beats-a_before != a_count*expected_tiles || b_model.beats-b_before != b_count)
        $fatal(1, "Wrong operand read count");
      if (s_model.beats-s_before != (fmt == 2 ? 0 : s_count)) $fatal(1, "Wrong scale read count");
      expected_issues = m*((k+R-1)/R)*expected_tiles;
      if (dut.u_ctrl.issued_slices != 64'(expected_issues))
        $fatal(1, "Wrong issue count");
      for (int i = 0; i < c_count; i++) if (!written[i]) $fatal(1, "C beat missing");
      $display("PASS engine case %0d: M=%0d K=%0d N=%0d fmt=%0d slices=%0d stalls=%0d",
               case_id, m, k, n, fmt, dut.u_ctrl.issued_slices, dut.u_ctrl.backpressure_cycles);
    end
    $fclose(fd);
    $display("PASS engine %0dx%0d: %0d commands, Python GEMM oracle", R, C, count);
    $finish;
  end
  initial begin #10000000; $fatal(1, "Engine timeout case=%0d state=%0d", case_id, dut.u_ctrl.state); end
endmodule
