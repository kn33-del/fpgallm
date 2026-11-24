// cocotb wrapper for T4: mem_subsys + ctrl_regs (flattened client ports, raw HBM AXI).
`include "tb_macros.svh"
module tb_t4 import llm_pkg::*; (
  input  logic clk, rst_n,

  // dma
  input  logic [$bits(dma_cmd_t)-1:0] dma_cmd,
  input  logic dma_cmd_valid,
  output logic dma_cmd_ready, dma_done,

  // client ports (Python acts as the client)
  `RD_SRV_PORTS(ifetch, BEAT_W),
  `RD_SRV_PORTS(mm_a, BEAT_W),
  `RD_SRV_PORTS(mm_s, BEAT_W),
  `RD_SRV_PORTS(mm_b, WSTREAM_W),
  `WR_SRV_PORTS(mm_c, BEAT_W),
  `RD_SRV_PORTS(attn_rd, BEAT_W),
  `WR_SRV_PORTS(attn_wr, BEAT_W),
  `WR_SRV_PORTS(kv, BEAT_W),
  `RD_SRV_PORTS(vec_rd, BEAT_W),
  `WR_SRV_PORTS(vec_wr, BEAT_W),

  // ctrl_regs AXI-Lite + sequencer side
  input  logic [11:0] s_axi_control_awaddr,
  input  logic        s_axi_control_awvalid,
  output logic        s_axi_control_awready,
  input  logic [31:0] s_axi_control_wdata,
  input  logic [3:0]  s_axi_control_wstrb,
  input  logic        s_axi_control_wvalid,
  output logic        s_axi_control_wready,
  output logic [1:0]  s_axi_control_bresp,
  output logic        s_axi_control_bvalid,
  input  logic        s_axi_control_bready,
  input  logic [11:0] s_axi_control_araddr,
  input  logic        s_axi_control_arvalid,
  output logic        s_axi_control_arready,
  output logic [31:0] s_axi_control_rdata,
  output logic [1:0]  s_axi_control_rresp,
  output logic        s_axi_control_rvalid,
  input  logic        s_axi_control_rready,
  output logic        interrupt,
  output logic        ap_start,
  input  logic        ap_done,
  output logic [$bits(addr_t)-1:0] instr_base,
  output logic [15:0] n_instr, token_pos,
  output logic [31:0] token_in,
  input  logic [31:0] result_token,
  output logic [$bits(perf_t)-1:0] perf,

  // HBM
  output logic [N_HBM-1:0][M_AXI_AW-1:0] m_axi_araddr,
  output logic [N_HBM-1:0][7:0]          m_axi_arlen,
  output logic [N_HBM-1:0][AXI_IDW-1:0]  m_axi_arid,
  output logic [N_HBM-1:0]               m_axi_arvalid,
  input  logic [N_HBM-1:0]               m_axi_arready,
  input  logic [N_HBM-1:0][AXI_DW-1:0]   m_axi_rdata,
  input  logic [N_HBM-1:0][AXI_IDW-1:0]  m_axi_rid,
  input  logic [N_HBM-1:0][1:0]          m_axi_rresp,
  input  logic [N_HBM-1:0]               m_axi_rlast,
  input  logic [N_HBM-1:0]               m_axi_rvalid,
  output logic [N_HBM-1:0]               m_axi_rready,
  output logic [N_HBM-1:0][M_AXI_AW-1:0] m_axi_awaddr,
  output logic [N_HBM-1:0][7:0]          m_axi_awlen,
  output logic [N_HBM-1:0][AXI_IDW-1:0]  m_axi_awid,
  output logic [N_HBM-1:0]               m_axi_awvalid,
  input  logic [N_HBM-1:0]               m_axi_awready,
  output logic [N_HBM-1:0][AXI_DW-1:0]   m_axi_wdata,
  output logic [N_HBM-1:0][AXI_DW/8-1:0] m_axi_wstrb,
  output logic [N_HBM-1:0]               m_axi_wlast,
  output logic [N_HBM-1:0]               m_axi_wvalid,
  input  logic [N_HBM-1:0]               m_axi_wready,
  input  logic [N_HBM-1:0][1:0]          m_axi_bresp,
  input  logic [N_HBM-1:0][AXI_IDW-1:0]  m_axi_bid,
  input  logic [N_HBM-1:0]               m_axi_bvalid,
  output logic [N_HBM-1:0]               m_axi_bready
);
  mem_rd_if ifetch_if (), mm_a_if (), mm_s_if (), attn_rd_if (), vec_rd_if ();
  mem_rd_if #(.W(WSTREAM_W)) mm_b_if ();
  mem_wr_if mm_c_if (), attn_wr_if (), kv_if (), vec_wr_if ();

  `RD_SRV_CONN(ifetch, ifetch_if)
  `RD_SRV_CONN(mm_a, mm_a_if)
  `RD_SRV_CONN(mm_s, mm_s_if)
  `RD_SRV_CONN(mm_b, mm_b_if)
  `WR_SRV_CONN(mm_c, mm_c_if)
  `RD_SRV_CONN(attn_rd, attn_rd_if)
  `WR_SRV_CONN(attn_wr, attn_wr_if)
  `WR_SRV_CONN(kv, kv_if)
  `RD_SRV_CONN(vec_rd, vec_rd_if)
  `WR_SRV_CONN(vec_wr, vec_wr_if)

  perf_t perf_s;
  logic  perf_clr;
  logic [N_HBM-1:0][63:0] hbm_base;
  addr_t instr_base_s;
  assign perf = perf_s;
  assign instr_base = instr_base_s;

  ctrl_regs u_regs (
    .clk, .rst_n,
    .awaddr (s_axi_control_awaddr), .awvalid (s_axi_control_awvalid), .awready (s_axi_control_awready),
    .wdata  (s_axi_control_wdata),  .wstrb (s_axi_control_wstrb), .wvalid (s_axi_control_wvalid),
    .wready (s_axi_control_wready), .bresp (s_axi_control_bresp), .bvalid (s_axi_control_bvalid),
    .bready (s_axi_control_bready), .araddr (s_axi_control_araddr), .arvalid (s_axi_control_arvalid),
    .arready(s_axi_control_arready), .rdata (s_axi_control_rdata), .rresp (s_axi_control_rresp),
    .rvalid (s_axi_control_rvalid), .rready (s_axi_control_rready), .interrupt,
    .ap_start, .ap_done, .instr_base (instr_base_s), .n_instr, .token_pos, .token_in, .result_token,
    .perf (perf_s), .perf_clr, .hbm_base
  );

  mem_subsys u_dut (
    .clk, .rst_n,
    .perf (perf_s), .perf_clr, .hbm_base,
    .dma_cmd (dma_cmd_t'(dma_cmd)), .dma_cmd_valid, .dma_cmd_ready, .dma_done,
    .ifetch_rd (ifetch_if.server), .mm_a_rd (mm_a_if.server), .mm_s_rd (mm_s_if.server),
    .mm_b_rd (mm_b_if.server), .mm_c_wr (mm_c_if.server),
    .attn_rd (attn_rd_if.server), .attn_wr (attn_wr_if.server), .kv_wr (kv_if.server),
    .vec_rd (vec_rd_if.server), .vec_wr (vec_wr_if.server),
    .m_axi_araddr, .m_axi_arlen, .m_axi_arid, .m_axi_arvalid, .m_axi_arready,
    .m_axi_rdata,  .m_axi_rid,   .m_axi_rresp, .m_axi_rlast, .m_axi_rvalid, .m_axi_rready,
    .m_axi_awaddr, .m_axi_awlen, .m_axi_awid, .m_axi_awvalid, .m_axi_awready,
    .m_axi_wdata,  .m_axi_wstrb, .m_axi_wlast, .m_axi_wvalid, .m_axi_wready,
    .m_axi_bresp,  .m_axi_bid,   .m_axi_bvalid, .m_axi_bready
  );
endmodule
