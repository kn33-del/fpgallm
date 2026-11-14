// =============================================================================
// TEAM 4 -- Memory + interfaces. Deliverable: mem_subsys serving every client port,
// plus ctrl_regs, plus the OFFLINE weight packer (python) that produces the layout contract.
//
// FIRST MILESTONE: weight_streamer + dummy consumer.
//   No board yet -> verify in sim against an AXI memory model with randomized read latency
//   (100-300 cycles), random per-PC stalls (models refresh), and back-pressure from consumer.
//   Pass criteria in sim: no deadlock, no FIFO overflow, lockstep reassembly bit-correct,
//   sustained >= 95% of beats/cycle when the model has no stalls.
//   On board bring-up: target PLAN_HBM_GBPS = 280 GB/s (see llm_pkg PLANNING BUDGET).
//
// Clocking (DECIDED for baseline): single 300 MHz domain. HBM AXI ports run at kernel clock.
//   Ceiling 307 GB/s. No async FIFO, no 450 MHz domain. Stretch goal only after baseline works.
// =============================================================================
module mem_subsys import llm_pkg::*; (
  input  logic clk, rst_n,

  input  dma_cmd_t  dma_cmd,
  input  logic      dma_cmd_valid,
  output logic      dma_cmd_ready,
  output logic      dma_done,

  mem_rd_if.server  ifetch_rd,
  mem_rd_if.server  mm_a_rd,
  mem_rd_if.server  mm_s_rd,
  mem_rd_if.server  mm_b_rd,     // wide
  mem_wr_if.server  mm_c_wr,
  mem_rd_if.server  attn_rd,
  mem_wr_if.server  attn_wr,
  mem_wr_if.server  kv_wr,
  mem_rd_if.server  vec_rd,
  mem_wr_if.server  vec_wr,

  output logic [N_HBM-1:0][AXI_AW-1:0]   m_axi_araddr,
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
  output logic [N_HBM-1:0][AXI_AW-1:0]   m_axi_awaddr,
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

  // ===========================================================================
  // Block diagram
  //
  //   mm_b_rd (wide) ----------------> weight_streamer --32 PC lanes--> hbm_port[0..31] -> m_axi
  //                                        ^ reassembly: 1 beat from EACH PC -> 1 tile
  //   narrow HBM traffic (ifetch, scales, gammas, embed dma, kv_wr, act spills)
  //        -> addr_router (rgn == HBM) -> narrow_hbm_arb -> hbm_port[pc(addr)] (shares AXI w/ streamer)
  //   ACT/SCRATCH traffic (mm_a, mm_c, attn_rd/wr, vec_rd/wr)
  //        -> addr_router (rgn == ACT/SCRATCH) -> act_buf (banked URAM)
  //   kv_wr -> kv_cache_mgr (layout transform) -> narrow HBM write path
  // ===========================================================================

  // ---- streamer <-> per-port AXI read channels (streamer owns AR/R on all 32 ports for
  //      the duration of a weight stream; narrow reads get priority-arbitrated in between)
  logic [N_HBM-1:0][AXI_AW-1:0] ws_araddr;  logic [N_HBM-1:0][7:0] ws_arlen;
  logic [N_HBM-1:0]             ws_arvalid, ws_arready;
  logic [N_HBM-1:0][AXI_DW-1:0] ws_rdata;   logic [N_HBM-1:0] ws_rvalid, ws_rready, ws_rlast;

  // ---- narrow HBM read/write engine <-> per-port AXI
  logic [N_HBM-1:0][AXI_AW-1:0] nr_araddr;  logic [N_HBM-1:0][7:0] nr_arlen;
  logic [N_HBM-1:0]             nr_arvalid, nr_arready;
  logic [N_HBM-1:0][AXI_DW-1:0] nr_rdata;   logic [N_HBM-1:0] nr_rvalid, nr_rready, nr_rlast;

  weight_streamer u_ws (
    .clk, .rst_n,
    .rd (mm_b_rd),
    .araddr (ws_araddr), .arlen (ws_arlen), .arvalid (ws_arvalid), .arready (ws_arready),
    .rdata  (ws_rdata),  .rvalid (ws_rvalid), .rready (ws_rready), .rlast (ws_rlast)
  );

  hbm_port_mux u_pmux (
    .clk, .rst_n
    // TODO: per port: merge ws_* and nr_* AR channels (use AXI ID bit to tag source),
    //   demux R by rid back to ws/nr. AW/W/B: only narrow engine writes. Drive m_axi_*.
  );

  narrow_hbm_engine u_nhe (
    .clk, .rst_n
    // TODO: serves HBM-region requests from ifetch, mm_s, attn/vec (gammas, rope tables),
    //   dma reads/writes, kv_cache_mgr writes. Maps addr -> (pc = addr[33:29], 512 MB per PC).
    //   Splits into AXI bursts: <=4 KB boundary, arlen<=255 (use 32-64 beat bursts).
  );

  act_buf u_act (
    .clk, .rst_n
    // TODO: serve ACT/SCRATCH requests from mm_a_rd, mm_c_wr, attn_rd/wr, vec_rd/wr, dma dst.
  );

  kv_cache_mgr u_kv (
    .clk, .rst_n
    // TODO: kv_wr in -> HBM writes in KV layout contract; forward to narrow_hbm_engine.
  );

  addr_router u_rt (
    .clk, .rst_n
    // TODO: for each narrow client: look at req.base.rgn, steer req + data to act_buf or
    //   narrow_hbm_engine; return data/done back to the right client. Clients never see this.
  );

  dma_engine u_dma (
    .clk, .rst_n
    // TODO: dma_cmd: read src (any rgn) -> write dst (any rgn). OP_EMBED = 1 row HBM->ACT.
  );

  // TODO: tie m_axi_* from u_pmux; tie dma_cmd_ready/dma_done from u_dma.

endmodule

// -----------------------------------------------------------------------------
// weight_streamer: THE critical block. One mem_req_t from mm_ctrl = whole matrix.
//   Per layout contract, PC r holds tile row r, contiguous per matrix -> each port reads one
//   long sequential range: base_pc + [0, n_tiles*k_tiles*32 B). No crossbar, no strides.
// -----------------------------------------------------------------------------
module weight_streamer import llm_pkg::*; (
  input  logic clk, rst_n,
  mem_rd_if.server rd,
  output logic [N_HBM-1:0][AXI_AW-1:0] araddr,
  output logic [N_HBM-1:0][7:0]        arlen,
  output logic [N_HBM-1:0]             arvalid,
  input  logic [N_HBM-1:0]             arready,
  input  logic [N_HBM-1:0][AXI_DW-1:0] rdata,
  input  logic [N_HBM-1:0]             rvalid,
  output logic [N_HBM-1:0]             rready,
  input  logic [N_HBM-1:0]             rlast
);
  // TODO:
  //  - 32 independent AR generators: long bursts (64 beats = 2 KB), keep >=16-32 outstanding per
  //    port (HBM latency is long; outstanding depth IS your bandwidth).
  //  - 32 per-PC R FIFOs (BRAM, deep: >= outstanding * burst). Lockstep pop: emit one 8192-b beat
  //    only when ALL 32 FIFOs non-empty. One slow PC (refresh) stalls everyone -> depth absorbs it.
  //  - Credit-based AR issue: never issue more AR than FIFO space (else R backs up into HBM ctrl).
  //  - b_is_kv requests use the same path (KV is in tile layout too).
  //  - Perf counters: bytes delivered, cycles rd.valid && !rd.ready (consumer stall),
  //    cycles !rd.valid while active (HBM starve). Export to ctrl_regs.
  //    Target: bytes / active_cycles >= 0.91 * 1024 B/cycle (= 280 GB/s at 300 MHz).
  //  - SLR: HBM + this block + PE array in SLR0. 8192-b bus must not cross SLRs.
endmodule

module hbm_port_mux      import llm_pkg::*; (input logic clk, rst_n); endmodule
module narrow_hbm_engine import llm_pkg::*; (input logic clk, rst_n); endmodule
module addr_router       import llm_pkg::*; (input logic clk, rst_n); endmodule
module dma_engine        import llm_pkg::*; (input logic clk, rst_n); endmodule

// -----------------------------------------------------------------------------
// act_buf: on-chip activation + scratch (URAM). Sizes (1B model, decode):
//   x residual D_MODEL*VEC_W = 4 KB; h int8 2 KB; qkv 3072*2B = 6 KB; scores 2048*2B = 4 KB;
//   gate/up 16384*2B = 32 KB; logits VOCAB*2B = 256 KB  <- biggest; still trivial vs ~34 MB URAM.
// Ports: mm_a read needs 1 beat/cycle SUSTAINED (feeds array). Give it a dedicated bank-port.
//   Concurrency in baseline: (mm_a rd + mm_c wr) OR (vec/attn rd + wr). URAM is 1R1W per bank ->
//   2 banks x (1R1W) + static routing by region is enough. Don't build a full crossbar.
// -----------------------------------------------------------------------------
module act_buf import llm_pkg::*; (input logic clk, rst_n);
  // TODO
endmodule

// -----------------------------------------------------------------------------
// kv_cache_mgr: places new K/V for (layer, kv_head, pos).
//   K^T layout: column write -> one byte per row-PC (wstrb byte enables). 32 tiny writes per head.
//   Baseline: accept the inefficiency (512 B/layer/token, irrelevant vs weights).
//   Later: buffer 32 positions on-chip, flush full beats.
//   Publish the address formula base(layer, kvh) to T2 in a header/doc.
// -----------------------------------------------------------------------------
module kv_cache_mgr import llm_pkg::*; (input logic clk, rst_n);
  // TODO
endmodule

// -----------------------------------------------------------------------------
// ctrl_regs: Vitis-standard AXI-Lite. Map:
//   0x00 ap_ctrl  [0]start [1]done(COR) [2]idle [3]ready   0x04 GIE  0x08 IER  0x0C ISR
//   0x10 instr_base_lo  0x14 instr_base_hi  0x18 n_instr  0x1C token_pos
//   0x20 result_token (RO)  0x40.. perf counters (RO)
// Must match kernel.xml arg offsets exactly, or XRT writes args into the void.
// -----------------------------------------------------------------------------
module ctrl_regs import llm_pkg::*; (
  input  logic clk, rst_n,
  input  logic [11:0] awaddr,  input  logic awvalid, output logic awready,
  input  logic [31:0] wdata,   input  logic [3:0] wstrb, input logic wvalid, output logic wready,
  output logic [1:0]  bresp,   output logic bvalid,  input  logic bready,
  input  logic [11:0] araddr,  input  logic arvalid, output logic arready,
  output logic [31:0] rdata,   output logic [1:0] rresp, output logic rvalid, input logic rready,
  output logic        interrupt,
  output logic        ap_start,
  input  logic        ap_done,
  output addr_t       instr_base,
  output logic [15:0] n_instr,
  output logic [15:0] token_pos,
  input  logic [31:0] result_token
);
  // TODO: standard AXI-Lite slave FSM. ap_start self-clears on ap_done (ap_ctrl_hs).
endmodule
