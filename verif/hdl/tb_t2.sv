// cocotb wrapper for T2 attn_unit (flattened interfaces).
`include "tb_macros.svh"
module tb_t2 import llm_pkg::*; (
  input  logic clk, rst_n,
  input  logic [$bits(attn_cmd_t)-1:0] cmd,
  input  logic cmd_valid,
  output logic cmd_ready, done,
  `MM_MST_PORTS(mm),
  `RD_CLI_PORTS(rd, BEAT_W),
  `WR_CLI_PORTS(wr, BEAT_W),
  `WR_CLI_PORTS(kv, BEAT_W)
);
  mm_if mm_if_i ();
  mem_rd_if rd_if ();
  mem_wr_if wr_if ();
  mem_wr_if kv_if ();

  `MM_MST_CONN(mm, mm_if_i)
  `RD_CLI_CONN(rd, rd_if)
  `WR_CLI_CONN(wr, wr_if)
  `WR_CLI_CONN(kv, kv_if)

  attn_unit u_dut (
    .clk, .rst_n,
    .cmd (attn_cmd_t'(cmd)), .cmd_valid, .cmd_ready, .done,
    .mm (mm_if_i.master), .rd (rd_if.client), .wr (wr_if.client), .kv (kv_if.client)
  );
endmodule
