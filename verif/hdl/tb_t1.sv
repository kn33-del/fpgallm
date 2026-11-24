// cocotb wrapper for T1 mm_engine (flattened interfaces).
`include "tb_macros.svh"
module tb_t1 import llm_pkg::*; #(
  parameter int REQUANT_SHIFT = 16
)(
  input logic clk, rst_n,
  `MM_SLV_PORTS(cmd),
  `RD_CLI_PORTS(a, BEAT_W),
  `RD_CLI_PORTS(s, BEAT_W),
  `RD_CLI_PORTS(b, WSTREAM_W),
  `WR_CLI_PORTS(c, BEAT_W)
);
  mm_if cmd_if ();
  mem_rd_if a_if ();
  mem_rd_if s_if ();
  mem_rd_if #(.W(WSTREAM_W)) b_if ();
  mem_wr_if c_if ();

  `MM_SLV_CONN(cmd, cmd_if)
  `RD_CLI_CONN(a, a_if)
  `RD_CLI_CONN(s, s_if)
  `RD_CLI_CONN(b, b_if)
  `WR_CLI_CONN(c, c_if)

  mm_engine #(.REQUANT_SHIFT(REQUANT_SHIFT)) u_dut (
    .clk, .rst_n,
    .cmd (cmd_if.slave),
    .a_rd (a_if.client), .s_rd (s_if.client), .b_rd (b_if.client),
    .c_wr (c_if.client)
  );
endmodule
