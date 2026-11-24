// =============================================================================
// llm_top.sv -- Vitis RTL kernel top. OWNER: integration.
//
// One decoder layer (decode, M=1), issued by llm_ctrl as macro-ops, strictly in order:
//   x : residual stream in ACT, VEC_W fixed point
//   1. OP_RMSNORM    (T3)  h = quant(rmsnorm(x) * g_attn)
//   2. OP_ATTN_BLOCK (T2)  a = Wo . attn(rope(Wq h), rope(Wk h), Wv h)   GEMMs on T1 via mm_arb
//   3. OP_RESADD     (T3)  x = x + a
//   4. OP_RMSNORM    (T3)  h = quant(rmsnorm(x) * g_mlp)
//   5. OP_MLP_BLOCK  (T3)  m = Wd . quant(silu(Wg h) .* (Wu h))         GEMMs on T1 via mm_arb
//   6. OP_RESADD     (T3)  x = x + m
// Start: OP_EMBED (T4 dma). End: OP_RMSNORM(final) -> OP_GEMM (LM head) -> OP_ARGMAX -> token.
//
// Who talks to whom:
//   ctrl_regs(T4) <-> llm_ctrl -> {mm_arb, attn_unit(T2), mlp_unit(T3), mem dma(T4)}
//   attn_unit, mlp_unit, llm_ctrl --mm_if--> mm_arb --> mm_engine(T1)
//   every unit --mem_rd_if/mem_wr_if--> mem_subsys(T4) --> 32x HBM AXI
// =============================================================================
module llm_top import llm_pkg::*; (
  input  logic ap_clk,
  input  logic ap_rst_n,

  // ---- s_axi_control (AXI-Lite). Vitis packaging needs exact names: generated wrapper flattens.
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

  // ---- 32x m_axi to HBM (one per PC). Wrapper flattens to m00_axi_*, m01_axi_*, ...
  //      Connectivity file (.cfg): sp=llm_top.m00_axi:HBM[0] ... m31_axi:HBM[31]
  output logic [N_HBM-1:0][M_AXI_AW-1:0] m_axi_araddr,
  output logic [N_HBM-1:0][7:0]         m_axi_arlen,
  output logic [N_HBM-1:0][AXI_IDW-1:0] m_axi_arid,
  output logic [N_HBM-1:0]              m_axi_arvalid,
  input  logic [N_HBM-1:0]              m_axi_arready,
  input  logic [N_HBM-1:0][AXI_DW-1:0]  m_axi_rdata,
  input  logic [N_HBM-1:0][AXI_IDW-1:0] m_axi_rid,
  input  logic [N_HBM-1:0][1:0]         m_axi_rresp,
  input  logic [N_HBM-1:0]              m_axi_rlast,
  input  logic [N_HBM-1:0]              m_axi_rvalid,
  output logic [N_HBM-1:0]              m_axi_rready,
  output logic [N_HBM-1:0][M_AXI_AW-1:0] m_axi_awaddr,
  output logic [N_HBM-1:0][7:0]         m_axi_awlen,
  output logic [N_HBM-1:0][AXI_IDW-1:0] m_axi_awid,
  output logic [N_HBM-1:0]              m_axi_awvalid,
  input  logic [N_HBM-1:0]              m_axi_awready,
  output logic [N_HBM-1:0][AXI_DW-1:0]  m_axi_wdata,
  output logic [N_HBM-1:0][AXI_DW/8-1:0] m_axi_wstrb,
  output logic [N_HBM-1:0]              m_axi_wlast,
  output logic [N_HBM-1:0]              m_axi_wvalid,
  input  logic [N_HBM-1:0]              m_axi_wready,
  input  logic [N_HBM-1:0][1:0]         m_axi_bresp,
  input  logic [N_HBM-1:0][AXI_IDW-1:0] m_axi_bid,
  input  logic [N_HBM-1:0]              m_axi_bvalid,
  output logic [N_HBM-1:0]              m_axi_bready
);

  logic clk, rst_n;
  assign clk   = ap_clk;
  assign rst_n = ap_rst_n;   // TODO(integration): register + replicate reset per SLR

  // ===========================================================================
  // Control registers (T4 owns ctrl_regs) <-> sequencer
  // ===========================================================================
  logic        ap_start, ap_done;
  addr_t       instr_base;
  logic [15:0] n_instr, token_pos;
  logic [31:0] token_in;
  perf_t       perf;
  logic [31:0] result_token;
  logic        perf_clr;
  logic [N_HBM-1:0][63:0] hbm_base;

  // ===========================================================================
  // Sequencer -> unit command channels
  // ===========================================================================
  attn_cmd_t attn_cmd;  logic attn_cmd_valid, attn_cmd_ready, attn_done;
  mlp_cmd_t  mlp_cmd;   logic mlp_cmd_valid,  mlp_cmd_ready,  mlp_done;
  vec_cmd_t  vec_cmd;   logic vec_cmd_valid,  vec_cmd_ready,  vec_done;
  logic [31:0] argmax_idx;                                   // valid when vec_done after V_ARGMAX
  dma_cmd_t  dma_cmd;   logic dma_cmd_valid,  dma_cmd_ready,  dma_done;

  // ===========================================================================
  // Matrix engine command fabric: 3 masters -> arbiter -> T1
  // ===========================================================================
  mm_if mm_from_ctrl ();   // LM head / generic OP_GEMM
  mm_if mm_from_attn ();   // QKV, QK^T, PV, Wo
  mm_if mm_from_mlp  ();   // gate/up, down
  mm_if mm_to_engine ();

  // ===========================================================================
  // Memory client ports (all served by T4 mem_subsys)
  // ===========================================================================
  mem_rd_if                  ifetch_rd ();   // ctrl: instruction fetch (HBM)
  mem_rd_if                  mm_a_rd   ();   // T1: activations, 1 beat = 32 int8 = one K-slice
  mem_rd_if                  mm_s_rd   ();   // T1: per-channel scales
  mem_rd_if #(.W(WSTREAM_W)) mm_b_rd   ();   // T1: weights/KV, 1 beat = one 32x32 tile
  mem_wr_if                  mm_c_wr   ();   // T1: results
  mem_rd_if                  attn_rd   ();   // T2: rope/softmax operands
  mem_wr_if                  attn_wr   ();   // T2: rope/softmax results
  mem_wr_if                  kv_wr     ();   // T2: new K/V for current pos -> KV cache
  mem_rd_if                  vec_rd    ();   // T3: vector unit operands (incl. gammas from HBM)
  mem_wr_if                  vec_wr    ();   // T3: vector unit results

  // ===========================================================================
  // Sequencer (integration-owned; assign an owner -- suggest T1 lead)
  // ===========================================================================
  llm_ctrl u_ctrl (
    .clk, .rst_n,
    .start (ap_start), .done (ap_done),
    .instr_base, .n_instr, .token_pos, .token_in, .result_token,
    .ifetch (ifetch_rd.client),
    .mm     (mm_from_ctrl.master),
    .attn_cmd, .attn_cmd_valid, .attn_cmd_ready, .attn_done,
    .mlp_cmd,  .mlp_cmd_valid,  .mlp_cmd_ready,  .mlp_done,
    .vec_cmd,  .vec_cmd_valid,  .vec_cmd_ready,  .vec_done, .argmax_idx,
    .dma_cmd,  .dma_cmd_valid,  .dma_cmd_ready,  .dma_done
  );

  // Arbiter: round-robin, one cmd in flight per master (baseline is in-order anyway).
  // Stamps tag[3:2] with master id; routes done back by done_tag[3:2].
  mm_arb u_mm_arb (
    .clk, .rst_n,
    .m0 (mm_from_ctrl.slave),
    .m1 (mm_from_attn.slave),
    .m2 (mm_from_mlp.slave),
    .s  (mm_to_engine.master)
  );

  // ===========================================================================
  // TEAM 1 -- compute
  // ===========================================================================
  mm_engine #(.REQUANT_SHIFT(T1_REQUANT_SHIFT)) u_mm (
    .clk, .rst_n,
    .cmd  (mm_to_engine.slave),
    .a_rd (mm_a_rd.client),
    .s_rd (mm_s_rd.client),
    .b_rd (mm_b_rd.client),
    .c_wr (mm_c_wr.client)
  );

  // ===========================================================================
  // TEAM 2 -- attention
  // ===========================================================================
  attn_unit u_attn (
    .clk, .rst_n,
    .cmd (attn_cmd), .cmd_valid (attn_cmd_valid), .cmd_ready (attn_cmd_ready), .done (attn_done),
    .mm  (mm_from_attn.master),
    .rd  (attn_rd.client),
    .wr  (attn_wr.client),
    .kv  (kv_wr.client)
  );

  // ===========================================================================
  // TEAM 3 -- MLP + vector/numerical units
  // ===========================================================================
  mlp_unit #(.QUANT_MULT(SCALE_W'(T3_QUANT_MULT)), .QUANT_SHIFT(T3_QUANT_SHIFT)) u_mlp (
    .clk, .rst_n,
    .mlp_cmd, .mlp_cmd_valid, .mlp_cmd_ready, .mlp_done,
    .vec_cmd, .vec_cmd_valid, .vec_cmd_ready, .vec_done, .argmax_idx,
    .mm (mm_from_mlp.master),
    .rd (vec_rd.client),
    .wr (vec_wr.client)
  );

  // ===========================================================================
  // TEAM 4 -- memory + interfaces
  // ===========================================================================
  ctrl_regs u_regs (
    .clk, .rst_n,
    .awaddr (s_axi_control_awaddr), .awvalid (s_axi_control_awvalid), .awready (s_axi_control_awready),
    .wdata  (s_axi_control_wdata),  .wstrb (s_axi_control_wstrb), .wvalid (s_axi_control_wvalid),
    .wready (s_axi_control_wready), .bresp (s_axi_control_bresp), .bvalid (s_axi_control_bvalid),
    .bready (s_axi_control_bready), .araddr (s_axi_control_araddr), .arvalid (s_axi_control_arvalid),
    .arready(s_axi_control_arready), .rdata (s_axi_control_rdata), .rresp (s_axi_control_rresp),
    .rvalid (s_axi_control_rvalid), .rready (s_axi_control_rready), .interrupt,
    .ap_start, .ap_done, .instr_base, .n_instr, .token_pos, .token_in, .result_token,
    .perf, .perf_clr, .hbm_base
  );

  mem_subsys u_mem (
    .clk, .rst_n,
    .perf, .perf_clr, .hbm_base,
    .dma_cmd, .dma_cmd_valid, .dma_cmd_ready, .dma_done,
    .ifetch_rd (ifetch_rd.server),
    .mm_a_rd   (mm_a_rd.server),
    .mm_s_rd   (mm_s_rd.server),
    .mm_b_rd   (mm_b_rd.server),
    .mm_c_wr   (mm_c_wr.server),
    .attn_rd   (attn_rd.server),
    .attn_wr   (attn_wr.server),
    .kv_wr     (kv_wr.server),
    .vec_rd    (vec_rd.server),
    .vec_wr    (vec_wr.server),
    .m_axi_araddr, .m_axi_arlen, .m_axi_arid, .m_axi_arvalid, .m_axi_arready,
    .m_axi_rdata,  .m_axi_rid,   .m_axi_rresp, .m_axi_rlast, .m_axi_rvalid, .m_axi_rready,
    .m_axi_awaddr, .m_axi_awlen, .m_axi_awid, .m_axi_awvalid, .m_axi_awready,
    .m_axi_wdata,  .m_axi_wstrb, .m_axi_wlast, .m_axi_wvalid, .m_axi_wready,
    .m_axi_bresp,  .m_axi_bid,   .m_axi_bvalid, .m_axi_bready
  );

endmodule


// =============================================================================
// mm_arb -- integration. 3 masters -> T1 engine.
// =============================================================================
module mm_arb import llm_pkg::*; (
  input logic clk, rst_n,
  mm_if.slave  m0, m1, m2,
  mm_if.master s
);
  //  - round-robin grant among {m0,m1,m2} cmd_valid
  //  - s.cmd = granted cmd with tag[3:2] = master index
  //  - s.cmd_valid/ready pass-through for granted master; others see cmd_ready=0
  //  - on s.done: route to master s.done_tag[3:2], give it done_tag with [3:2] cleared
  //  - baseline: only one outstanding cmd total (T1 engine is single-issue)
  //  - grant is locked while s.cmd_valid && !s.cmd_ready (payload stable per llm_if rules)
  logic [2:0] req;
  logic [1:0] last_gnt, gnt, gnt_q;
  logic       gnt_v, busy, lock;
  mm_cmd_t    sel;

  assign req = {m2.cmd_valid, m1.cmd_valid, m0.cmd_valid};

  always_comb begin
    gnt   = gnt_q;
    gnt_v = lock;
    if (!lock) begin
      for (int i = 3; i >= 1; i--) begin          // first requester after last_gnt wins
        automatic int c = (int'(last_gnt) + i) % 3;
        if (req[c]) begin gnt = 2'(c); gnt_v = 1'b1; end
      end
    end
    case (gnt)
      2'd1:    sel = m1.cmd;
      2'd2:    sel = m2.cmd;
      default: sel = m0.cmd;
    endcase
    sel.tag[3:2] = gnt;
  end

  assign s.cmd       = sel;
  assign s.cmd_valid = !busy && gnt_v;
  assign m0.cmd_ready = !busy && gnt_v && gnt == 2'd0 && s.cmd_ready;
  assign m1.cmd_ready = !busy && gnt_v && gnt == 2'd1 && s.cmd_ready;
  assign m2.cmd_ready = !busy && gnt_v && gnt == 2'd2 && s.cmd_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0; lock <= 1'b0; last_gnt <= 2'd2; gnt_q <= '0;
    end else begin
      lock  <= s.cmd_valid && !s.cmd_ready;
      gnt_q <= gnt;
      if (s.cmd_valid && s.cmd_ready) begin
        busy     <= 1'b1;
        last_gnt <= gnt;
      end
      if (s.done) busy <= 1'b0;
    end
  end

  always_comb begin
    m0.done = s.done && s.done_tag[3:2] == 2'd0;
    m1.done = s.done && s.done_tag[3:2] == 2'd1;
    m2.done = s.done && s.done_tag[3:2] == 2'd2;
    m0.done_tag = {2'b00, s.done_tag[1:0]};
    m1.done_tag = {2'b00, s.done_tag[1:0]};
    m2.done_tag = {2'b00, s.done_tag[1:0]};
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk) if (rst_n && s.done)
    assert (busy && s.done_tag[3:2] != 2'd3) else $error("mm_arb: unexpected done (tag %h)", s.done_tag);
`endif
endmodule
