`timescale 1ns/1ps
module tb_t1_array import llm_pkg::*; #(
  parameter int R = 3, C = 5
);
  logic clk = 0;
  always #5 clk = !clk;
  logic rst_n = 0, stall = 0, a_valid = 0, w_load = 0;
  logic [R-1:0][DATA_W-1:0] a;
  logic [R-1:0][C-1:0][DATA_W-1:0] w;
  logic [C-1:0][ACC_W-1:0] psum;
  logic valid;
  logic [C-1:0][ACC_W-1:0] expected[$];
  int signed weights[R][C];
  int accepted = 0, checked = 0;
  logic [31:0] rng = 32'h2917abcd;
  systolic_array #(.R(R), .C(C)) dut (
    .clk, .rst_n, .a_in(a), .a_valid, .w_in(w), .w_load, .stall,
    .psum_out(psum), .psum_valid(valid)
  );
  always @(posedge clk) begin : scoreboard
    logic [C-1:0][ACC_W-1:0] result, want;
    int signed sum;
    if (!rst_n) begin
      expected.delete();
      for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) weights[r][c] = 0;
    end else if (!stall) begin
      if (valid) begin
        if (expected.size() == 0) $fatal(1, "Unexpected array output");
        want = expected.pop_front();
        if (psum !== want) $fatal(1, "Array mismatch got=%h expected=%h", psum, want);
        checked++;
      end
      if (w_load) for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) weights[r][c] = int'($signed(w[r][c]));
      if (a_valid) begin
        for (int c = 0; c < C; c++) begin
          sum = 0;
          for (int r = 0; r < R; r++) sum += int'($signed(a[r])) * weights[r][c];
          result[c] = ACC_W'(sum);
        end
        expected.push_back(result);
        accepted++;
      end
    end
  end
  initial begin
    a = '0;
    w = '0;
    repeat (4) @(negedge clk);
    rst_n = 1;
    for (int t = 0; t < 700; t++) begin
      @(negedge clk);
      rng = {rng[30:0], rng[31]^rng[21]^rng[1]^rng[0]};
      stall = t < 100 ? 0 : rng[3:2] == 0;
      a_valid = t < 100 ? 1 : rng[5:4] != 0;
      w_load = t < 100 ? 1 : rng[7:6] != 0;
      for (int r = 0; r < R; r++) begin
        a[r] = DATA_W'(t*17 + r*43);
        for (int c = 0; c < C; c++) w[r][c] = DATA_W'(t*31+r*11+c*37);
      end
    end
    @(negedge clk);
    a_valid = 0;
    w_load = 0;
    stall = 0;
    repeat (R+C+5) @(negedge clk);
    if (expected.size() != 0 || checked != accepted)
      $fatal(1, "Array failed to drain: %0d/%0d", checked, accepted);
    // Reset while data is in flight. Old valid bits must never reappear.
    a_valid = 1;
    w_load = 1;
    repeat (2) @(negedge clk);
    rst_n = 0;
    a_valid = 0;
    w_load = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    repeat (R+C+5) @(negedge clk);
    if (valid) $fatal(1, "Output leaked across reset");
    $display("PASS array %0dx%0d: %0d outputs, stalls/reuse/bubbles/reset", R, C, checked);
    $finish;
  end
  initial begin #1000000; $fatal(1, "Array timeout"); end
endmodule
