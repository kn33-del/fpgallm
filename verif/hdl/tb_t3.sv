// cocotb wrapper for T3 mlp_unit (+ vec_unit) (flattened interfaces).
`include "tb_macros.svh"
module tb_t3 import llm_pkg::*; (
  input  logic clk, rst_n,
  input  logic [$bits(mlp_cmd_t)-1:0] mlp_cmd,
  input  logic mlp_cmd_valid,
  output logic mlp_cmd_ready, mlp_done,
  input  logic [$bits(vec_cmd_t)-1:0] vec_cmd,
  input  logic vec_cmd_valid,
  output logic vec_cmd_ready, vec_done,
  output logic [31:0] argmax_idx,
  `MM_MST_PORTS(mm),
  `RD_CLI_PORTS(rd, BEAT_W),
  `WR_CLI_PORTS(wr, BEAT_W)
);
  mm_if mm_if_i ();
  mem_rd_if rd_if ();
  mem_wr_if wr_if ();

  `MM_MST_CONN(mm, mm_if_i)
  `RD_CLI_CONN(rd, rd_if)
  `WR_CLI_CONN(wr, wr_if)

  mlp_unit u_dut (
    .clk, .rst_n,
    .mlp_cmd (mlp_cmd_t'(mlp_cmd)), .mlp_cmd_valid, .mlp_cmd_ready, .mlp_done,
    .vec_cmd (vec_cmd_t'(vec_cmd)), .vec_cmd_valid, .vec_cmd_ready, .vec_done, .argmax_idx,
    .mm (mm_if_i.master), .rd (rd_if.client), .wr (wr_if.client)
  );
endmodule
