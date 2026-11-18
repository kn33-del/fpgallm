`timescale 1ns/1ps
module tb_t1_requant import llm_pkg::*; #(
  parameter int C = 32, SHIFT = 16
);
  logic clk = 0;
  always #5 clk = !clk;
  logic rst_n = 0;
  logic [C-1:0][ACC_W-1:0] acc;
  logic [C-1:0][SCALE_W-1:0] scales;
  logic acc_valid, acc_ready, acc_last, scales_valid;
  mm_out_e fmt;
  logic [BEAT_W-1:0] data;
  logic valid, ready, last;
  logic [BEAT_W-1:0] expected [1024];
  logic expected_last [1024];
  logic [C-1:0][ACC_W-1:0] inputs [128];
  logic [C-1:0][SCALE_W-1:0] factors [128];
  mm_out_e formats [128];
  logic input_last [128];
  int checked = 0, cycles = 0, total_outputs;
  mm_requant #(.C(C), .SHIFT(SHIFT)) dut (
    .clk, .rst_n, .acc, .acc_valid, .acc_ready, .acc_last,
    .scales, .scales_valid, .fmt,
    .wr_data(data), .wr_valid(valid), .wr_ready(ready), .wr_last(last)
  );
  assign ready = cycles % 71 < 31 && cycles % 5 != 0;
  always @(posedge clk) begin
    cycles <= cycles + 1;
    if (rst_n && valid && ready) begin
      if (checked >= total_outputs || data !== expected[checked] || last != expected_last[checked])
        $fatal(1, "Requant SHIFT=%0d beat=%0d got=%h last=%b expected=%h last=%b",
               SHIFT, checked, data, last, expected[checked], expected_last[checked]);
      checked++;
    end
  end
  assert property (@(posedge clk) disable iff (!rst_n)
    valid && !ready |=> valid && $stable({data,last}));
  initial begin
    string path;
    int fd, scan, total_inputs, f, l;
    acc = '0; scales = '0; acc_valid = 0; acc_last = 0;
    scales_valid = 0; fmt = OUT_I8;
    if (!$value$plusargs("vectors=%s", path)) $fatal(1, "Missing vectors");
    fd = $fopen(path, "r");
    if (fd == 0) $fatal(1, "Cannot read requant vectors");
    scan = $fscanf(fd, "%d %d", total_inputs, total_outputs);
    for (int i = 0; i < total_inputs; i++) begin
      scan = $fscanf(fd, "%d %d %h %h", f, l, inputs[i], factors[i]);
      formats[i] = mm_out_e'(f);
      input_last[i] = 1'(l);
    end
    for (int i = 0; i < total_outputs; i++) begin
      scan = $fscanf(fd, "%h %d", expected[i], l);
      expected_last[i] = 1'(l);
    end
    $fclose(fd);
    repeat (4) @(negedge clk);
    rst_n = 1;
    for (int i = 0; i < total_inputs; i++) begin
      @(negedge clk);
      acc = inputs[i]; scales = factors[i]; fmt = formats[i]; acc_last = input_last[i];
      acc_valid = 1;
      // A delayed scale-valid tests that a held input is not accepted early.
      scales_valid = (i % 7 != 0);
      if (!scales_valid && fmt != OUT_ACC32) begin
        repeat (3) begin
          @(posedge clk);
          if (acc_ready) $fatal(1, "Accepted input without scales");
          @(negedge clk);
        end
        scales_valid = 1;
      end
      do @(posedge clk); while (!acc_ready);
      @(negedge clk);
      acc_valid = 0;
    end
    do @(negedge clk); while (checked < total_outputs);
    repeat (8) @(negedge clk);
    if (valid) $fatal(1, "Extra requant output");
    $display("PASS requant SHIFT=%0d: %0d vectors / %0d beats, INT32 extremes and mixed formats",
             SHIFT, total_inputs, checked);
    $finish;
  end
  initial begin #1000000; $fatal(1, "Requant timeout"); end
endmodule
