// =============================================================================
// TEAM 4 -- Memory + interfaces.
//   mem_subsys       : top of T4. Wires everything below to the client ports + 32 HBM AXI ports.
//   weight_streamer  : wide 8192-b tile stream from all 32 PCs (the bandwidth-critical block)
//   hbm_port_mux     : per AXI port, shares AR/R between streamer and narrow engine (ID-tagged)
//   narrow_hbm_engine: small HBM reads/writes (ifetch, scales, gammas, dma, kv) -> AXI bursts
//   addr_router      : steers each narrow client to act_buf or HBM by addr_t.rgn
//   act_buf          : on-chip ACT + SCRATCH (URAM), 1 read port + 1 write port
//   kv_cache_mgr     : writes new K/V vectors into the KV layout contract (llm_pkg)
//   dma_engine       : copy any region -> any region (OP_EMBED)
//   ctrl_regs        : Vitis AXI-Lite control/status + perf counters
//   t4_fifo          : FWFT FIFO (sync-read RAM + output reg -> BRAM/URAM friendly)
//
// Baseline decisions (see llm_pkg PLANNING BUDGET):
//   - single 300 MHz domain, HBM ports at kernel clock, ceiling 307 GB/s, plan 280 GB/s
//   - narrow traffic is simple and serialized (one read + one write request in flight).
//     It's <1% of bytes; don't optimize it before the streamer hits target on hardware.
//   - all requests must be 32-B aligned (beat aligned). Byte-granular writes only via kv path.
//   - narrow HBM reads: a request may span PCs, but a new PC is only started after the
//     previous PC's bursts drained (keeps R in order). Keep narrow data inside one PC.
// =============================================================================


// -----------------------------------------------------------------------------
// t4_fifo: first-word-fall-through FIFO. Capacity = DEPTH + 1 (output register).
// -----------------------------------------------------------------------------
module t4_fifo #(
  parameter int W     = 8,
  parameter int DEPTH = 16                       // power of 2
)(
  input  logic                     clk, rst_n,
  input  logic [W-1:0]             din,
  input  logic                     push,         // ignored when full
  output logic                     full,
  output logic [W-1:0]             dout,
  output logic                     valid,
  input  logic                     pop,          // only when valid
  output logic [$clog2(DEPTH)+1:0] count
);
  localparam int AW = $clog2(DEPTH);
  localparam logic [AW:0] DEP = (AW+1)'(DEPTH);

  logic [W-1:0] mem [DEPTH];
  logic [AW:0]  wptr, rptr;
  logic         mem_empty, fetch;

  assign mem_empty = (wptr == rptr);
  assign full      = ((wptr - rptr) == DEP);
  assign fetch     = !mem_empty && (!valid || pop);
  assign count     = (AW+2)'(wptr - rptr) + (AW+2)'(valid);

  always_ff @(posedge clk) begin
    if (push && !full) mem[wptr[AW-1:0]] <= din;
    if (fetch)         dout <= mem[rptr[AW-1:0]];
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wptr <= '0; rptr <= '0; valid <= 1'b0;
    end else begin
      if (push && !full) wptr <= wptr + 1'b1;
      if (fetch)         rptr <= rptr + 1'b1;
      if (fetch)         valid <= 1'b1;
      else if (pop)      valid <= 1'b0;
    end
  end
endmodule


// =============================================================================
// mem_subsys
// =============================================================================
module mem_subsys import llm_pkg::*; (
  input  logic clk, rst_n,

  input  logic      perf_clr,      // pulse (ctrl_regs pulses it on every ap_start)
  output perf_t     perf,
  // Per-port HBM buffer base (host pointer args). Kernel address {pc, off} -> port pc, base[pc]+off.
  input  logic [N_HBM-1:0][63:0] hbm_base,

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

  // ===========================================================================
  // Block diagram
  //   mm_b_rd (wide) --> weight_streamer --ws_*--> hbm_port_mux --> m_axi[0..31]
  //   narrow rd clients --+                      ^
  //   narrow wr clients --+--> addr_router --+--> narrow_hbm_engine --nr_*--+
  //                                          +--> act_buf (URAM)
  //   kv_wr --> kv_cache_mgr --(byte-strobe ops)--> narrow_hbm_engine
  //   dma_cmd --> dma_engine (is itself one rd client + one wr client of the router)
  // ===========================================================================

  localparam int NRD = 6;   // 0 ifetch, 1 mm_a, 2 mm_s, 3 attn, 4 vec, 5 dma   (low idx = priority)
  localparam int NWR = 4;   // 0 mm_c, 1 attn, 2 vec, 3 dma

  // ---- streamer <-> port mux
  logic [N_HBM-1:0][AXI_AW-1:0] ws_araddr;  logic [N_HBM-1:0][7:0] ws_arlen;
  logic [N_HBM-1:0]             ws_arvalid, ws_arready;
  logic [N_HBM-1:0][AXI_DW-1:0] ws_rdata;   logic [N_HBM-1:0] ws_rvalid, ws_rready, ws_rlast;

  // ---- narrow engine <-> port mux
  logic [N_HBM-1:0][AXI_AW-1:0]   nr_araddr, nr_awaddr;
  logic [N_HBM-1:0][7:0]          nr_arlen,  nr_awlen;
  logic [N_HBM-1:0]               nr_arvalid, nr_arready, nr_awvalid, nr_awready;
  logic [N_HBM-1:0][AXI_DW-1:0]   nr_rdata,  nr_wdata;
  logic [N_HBM-1:0][AXI_DW/8-1:0] nr_wstrb;
  logic [N_HBM-1:0]               nr_rvalid, nr_rready, nr_rlast;
  logic [N_HBM-1:0]               nr_wvalid, nr_wready, nr_wlast;
  logic [N_HBM-1:0]               nr_bvalid, nr_bready;

  // ---- router client arrays
  mem_req_t [NRD-1:0]             c_rq;
  logic     [NRD-1:0]             c_rq_v, c_rq_r, c_rv, c_rl, c_rr;
  logic     [NRD-1:0][BEAT_W-1:0] c_rd;
  mem_req_t [NWR-1:0]             c_wq;
  logic     [NWR-1:0]             c_wq_v, c_wq_r, c_wv, c_wl, c_wr, c_wdone;
  logic     [NWR-1:0][BEAT_W-1:0] c_wd;

  // ---- router server arrays: index 0 = act_buf, 1 = narrow HBM
  mem_req_t [1:0]                 s_rq, s_wq;
  logic     [1:0]                 s_rq_v, s_rq_r, s_rv, s_rl, s_rr;
  logic     [1:0][BEAT_W-1:0]     s_rd, s_wd;
  logic     [1:0]                 s_wq_v, s_wq_r, s_wv, s_wl, s_wr, s_wdone;

  // ---- kv path
  logic                kv_hold, kv_op_v, kv_op_r, w_quiet;
  logic [AXI_AW-1:0]   kv_op_addr;
  logic [AXI_DW-1:0]   kv_op_data;
  logic [AXI_DW/8-1:0] kv_op_strb;

  // ---- dma client signals
  mem_req_t            d_rq, d_wq;
  logic                d_rq_v, d_rq_r, d_rv, d_rl, d_rr;
  logic                d_wq_v, d_wq_r, d_wv, d_wl, d_wr, d_wdone;
  logic [BEAT_W-1:0]   d_rd, d_wd;

  // ===========================================================================
  // Interface -> array glue
  // ===========================================================================
  `define T4_RD_CLIENT(IDX, IF) \
    assign c_rq[IDX]   = IF.req;        assign c_rq_v[IDX] = IF.req_valid; \
    assign IF.req_ready = c_rq_r[IDX];  assign IF.data     = c_rd[IDX];    \
    assign IF.valid    = c_rv[IDX];     assign IF.last     = c_rl[IDX];    \
    assign c_rr[IDX]   = IF.ready;

  `define T4_WR_CLIENT(IDX, IF) \
    assign c_wq[IDX]   = IF.req;        assign c_wq_v[IDX] = IF.req_valid; \
    assign IF.req_ready = c_wq_r[IDX];  assign c_wd[IDX]   = IF.data;      \
    assign c_wv[IDX]   = IF.valid;      assign c_wl[IDX]   = IF.last;      \
    assign IF.ready    = c_wr[IDX];     assign IF.done     = c_wdone[IDX];

  `T4_RD_CLIENT(0, ifetch_rd)
  `T4_RD_CLIENT(1, mm_a_rd)
  `T4_RD_CLIENT(2, mm_s_rd)
  `T4_RD_CLIENT(3, attn_rd)
  `T4_RD_CLIENT(4, vec_rd)
  `T4_WR_CLIENT(0, mm_c_wr)
  `T4_WR_CLIENT(1, attn_wr)
  `T4_WR_CLIENT(2, vec_wr)

  `undef T4_RD_CLIENT
  `undef T4_WR_CLIENT

  assign c_rq[5] = d_rq;  assign c_rq_v[5] = d_rq_v;  assign d_rq_r = c_rq_r[5];
  assign d_rd = c_rd[5];  assign d_rv = c_rv[5];      assign d_rl = c_rl[5];  assign c_rr[5] = d_rr;
  assign c_wq[3] = d_wq;  assign c_wq_v[3] = d_wq_v;  assign d_wq_r = c_wq_r[3];
  assign c_wd[3] = d_wd;  assign c_wv[3] = d_wv;      assign c_wl[3] = d_wl;
  assign d_wr = c_wr[3];  assign d_wdone = c_wdone[3];

  // ===========================================================================
  // Blocks
  // ===========================================================================
  weight_streamer u_ws (
    .clk, .rst_n,
    .rd (mm_b_rd),
    .araddr (ws_araddr), .arlen (ws_arlen), .arvalid (ws_arvalid), .arready (ws_arready),
    .rdata  (ws_rdata),  .rvalid (ws_rvalid), .rready (ws_rready), .rlast (ws_rlast),
    .perf_clr,
    .p_beats (perf.ws_beats), .p_active (perf.ws_active),
    .p_starve (perf.ws_starve), .p_stall (perf.ws_stall)
  );

  hbm_port_mux u_pmux (
    .clk, .rst_n, .err_clr (perf_clr), .axi_err (perf.axi_err), .hbm_base,
    .ws_araddr, .ws_arlen, .ws_arvalid, .ws_arready, .ws_rdata, .ws_rvalid, .ws_rlast, .ws_rready,
    .nr_araddr, .nr_arlen, .nr_arvalid, .nr_arready, .nr_rdata, .nr_rvalid, .nr_rlast, .nr_rready,
    .nr_awaddr, .nr_awlen, .nr_awvalid, .nr_awready,
    .nr_wdata, .nr_wstrb, .nr_wlast, .nr_wvalid, .nr_wready,
    .nr_bvalid, .nr_bready,
    .m_axi_araddr, .m_axi_arlen, .m_axi_arid, .m_axi_arvalid, .m_axi_arready,
    .m_axi_rdata,  .m_axi_rid,   .m_axi_rresp, .m_axi_rlast, .m_axi_rvalid, .m_axi_rready,
    .m_axi_awaddr, .m_axi_awlen, .m_axi_awid, .m_axi_awvalid, .m_axi_awready,
    .m_axi_wdata,  .m_axi_wstrb, .m_axi_wlast, .m_axi_wvalid, .m_axi_wready,
    .m_axi_bresp,  .m_axi_bid,   .m_axi_bvalid, .m_axi_bready
  );

  narrow_hbm_engine u_nhe (
    .clk, .rst_n,
    .rq (s_rq[1]), .rq_v (s_rq_v[1]), .rq_r (s_rq_r[1]),
    .rdata (s_rd[1]), .rv (s_rv[1]), .rl (s_rl[1]), .rr (s_rr[1]),
    .wq (s_wq[1]), .wq_v (s_wq_v[1]), .wq_r (s_wq_r[1]),
    .wdata (s_wd[1]), .wv (s_wv[1]), .wl (s_wl[1]), .wr (s_wr[1]), .wdone (s_wdone[1]),
    .kv_hold, .kv_addr (kv_op_addr), .kv_data (kv_op_data), .kv_strb (kv_op_strb),
    .kv_v (kv_op_v), .kv_r (kv_op_r), .w_quiet,
    .nr_araddr, .nr_arlen, .nr_arvalid, .nr_arready, .nr_rdata, .nr_rvalid, .nr_rlast, .nr_rready,
    .nr_awaddr, .nr_awlen, .nr_awvalid, .nr_awready,
    .nr_wdata, .nr_wstrb, .nr_wlast, .nr_wvalid, .nr_wready,
    .nr_bvalid, .nr_bready
  );

  act_buf u_act (
    .clk, .rst_n,
    .rq (s_rq[0]), .rq_v (s_rq_v[0]), .rq_r (s_rq_r[0]),
    .rdata (s_rd[0]), .rv (s_rv[0]), .rl (s_rl[0]), .rr (s_rr[0]),
    .wq (s_wq[0]), .wq_v (s_wq_v[0]), .wq_r (s_wq_r[0]),
    .wdata (s_wd[0]), .wv (s_wv[0]), .wl (s_wl[0]), .wr (s_wr[0]), .wdone (s_wdone[0])
  );

  kv_cache_mgr u_kv (
    .clk, .rst_n,
    .kv (kv_wr),
    .kv_hold, .op_addr (kv_op_addr), .op_data (kv_op_data), .op_strb (kv_op_strb),
    .op_v (kv_op_v), .op_r (kv_op_r), .w_quiet
  );

  addr_router #(.NRD(NRD), .NWR(NWR)) u_rt (
    .clk, .rst_n,
    .c_rq, .c_rq_v, .c_rq_r, .c_rd, .c_rv, .c_rl, .c_rr,
    .c_wq, .c_wq_v, .c_wq_r, .c_wd, .c_wv, .c_wl, .c_wr, .c_wdone,
    .s_rq, .s_rq_v, .s_rq_r, .s_rd, .s_rv, .s_rl, .s_rr,
    .s_wq, .s_wq_v, .s_wq_r, .s_wd, .s_wv, .s_wl, .s_wr, .s_wdone
  );

  dma_engine u_dma (
    .clk, .rst_n,
    .cmd (dma_cmd), .cmd_v (dma_cmd_valid), .cmd_r (dma_cmd_ready), .done (dma_done),
    .rq (d_rq), .rq_v (d_rq_v), .rq_r (d_rq_r), .rd (d_rd), .rv (d_rv), .rl (d_rl), .rr (d_rr),
    .wq (d_wq), .wq_v (d_wq_v), .wq_r (d_wq_r), .wd (d_wd), .wv (d_wv), .wl (d_wl), .wr (d_wr),
    .wdone (d_wdone)
  );

endmodule


// =============================================================================
// weight_streamer: THE critical block.
//   One mem_req_t from mm_ctrl = whole matrix (2D: row = one n_tile's k_tiles).
//   Address in each PC p: {p, base.off[28:0] + row*row_stride + col*32}. base.off[33:29] ignored.
//   Per PC: independent AR generator, credit-limited by its R FIFO, so R never backs up into HBM.
//   Output: one 8192-b beat when ALL 32 FIFOs have data. PC r -> data[r*256 +: 256] = tile row r.
// =============================================================================
module weight_streamer import llm_pkg::*; #(
  parameter int BURST  = 64,     // beats per AR (2 KB). Must be <= 128 (4 KB) and <= 256.
  parameter int FDEPTH = 512     // per-PC FIFO beats. >= latency_cycles + a few bursts.
)(
  input  logic clk, rst_n,
  mem_rd_if.server rd,
  output logic [N_HBM-1:0][AXI_AW-1:0] araddr,
  output logic [N_HBM-1:0][7:0]        arlen,
  output logic [N_HBM-1:0]             arvalid,
  input  logic [N_HBM-1:0]             arready,
  input  logic [N_HBM-1:0][AXI_DW-1:0] rdata,
  input  logic [N_HBM-1:0]             rvalid,
  output logic [N_HBM-1:0]             rready,
  input  logic [N_HBM-1:0]             rlast,
  input  logic                         perf_clr,
  output logic [47:0]                  p_beats, p_active, p_starve, p_stall
);
  localparam int CW = $clog2(FDEPTH) + 2;

  // ---- latched request
  logic        active;
  logic [15:0] row_beats, n_rows;
  logic [28:0] stride;
  logic [31:0] total, out_cnt;
  logic        start, pop;

  assign start        = rd.req_valid && rd.req_ready;
  assign rd.req_ready = !active;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      active <= 1'b0; out_cnt <= '0; total <= '0;
      row_beats <= '0; n_rows <= '0; stride <= '0;
    end else if (start) begin
      active    <= (rd.req.row_beats != 0) && (rd.req.n_rows != 0);
      row_beats <= rd.req.row_beats;
      n_rows    <= rd.req.n_rows;
      stride    <= rd.req.row_stride[28:0];
      total     <= 32'(rd.req.row_beats) * 32'(rd.req.n_rows);
      out_cnt   <= '0;
    end else if (pop) begin
      out_cnt <= out_cnt + 1;
      if (out_cnt == total - 1) active <= 1'b0;
    end
  end

  // ---- per-PC AR generators + FIFOs
  logic [N_HBM-1:0][AXI_DW-1:0] f_dout;
  logic [N_HBM-1:0]             f_valid, f_full;

  for (genvar p = 0; p < N_HBM; p++) begin : g_pc
    logic [15:0]   col, row;
    logic [28:0]   row_addr, a;
    logic          ar_done;
    logic [CW-1:0] credit;
    logic [15:0]   left, to4k, len;
    logic          ar_hs;

    always_comb begin
      a    = row_addr + {8'd0, col, 5'd0};
      left = row_beats - col;
      to4k = 16'd128 - {9'd0, a[11:5]};
      len  = 16'(BURST);
      if (left < len) len = left;
      if (to4k < len) len = to4k;
    end

    assign araddr[p]  = {5'(p), a};
    assign arlen[p]   = 8'(len - 16'd1);
    assign arvalid[p] = active && !ar_done && (credit >= CW'(len));
    assign ar_hs      = arvalid[p] && arready[p];

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        col <= '0; row <= '0; row_addr <= '0; ar_done <= 1'b1;
      end else if (start) begin
        col <= '0; row <= '0; row_addr <= rd.req.base.off[28:0]; ar_done <= 1'b0;
      end else if (ar_hs) begin
        if (col + len == row_beats) begin
          col      <= '0;
          row      <= row + 1;
          row_addr <= row_addr + stride;
          if (row + 1 == n_rows) ar_done <= 1'b1;
        end else begin
          col <= col + len;
        end
      end
    end

    // credit = free FIFO space not yet promised to an outstanding AR
    always_ff @(posedge clk) begin
      if (!rst_n) credit <= CW'(FDEPTH);
      else        credit <= credit - (ar_hs ? CW'(len) : '0) + (pop ? CW'(1) : '0);
    end

    t4_fifo #(.W(AXI_DW), .DEPTH(FDEPTH)) u_f (
      .clk, .rst_n,
      .din (rdata[p]), .push (rvalid[p] && rready[p]), .full (f_full[p]),
      .dout (f_dout[p]), .valid (f_valid[p]), .pop (pop), .count ()
    );
    assign rready[p] = !f_full[p];   // never actually full: credits guarantee space
  end

  // ---- lockstep output
  assign rd.valid = active && (&f_valid);
  assign rd.data  = f_dout;
  assign rd.last  = (out_cnt == total - 1);
  assign pop      = rd.valid && rd.ready;

  // ---- perf counters
  always_ff @(posedge clk) begin
    if (!rst_n || perf_clr) begin
      p_beats <= '0; p_active <= '0; p_starve <= '0; p_stall <= '0;
    end else begin
      if (pop)                    p_beats  <= p_beats + 1;
      if (active)                 p_active <= p_active + 1;
      if (active && !rd.valid)    p_starve <= p_starve + 1;
      if (rd.valid && !rd.ready)  p_stall  <= p_stall + 1;
    end
  end

  // rlast unused: bursts are tracked by credits, not by count
  logic unused_rlast;
  assign unused_rlast = ^rlast;
endmodule


// =============================================================================
// hbm_port_mux: per AXI port.
//   AR: narrow has priority (tiny, latency-sensitive); grant is held while arvalid && !arready.
//   ID MSB tags source (0 = streamer, 1 = narrow); R demuxed by rid MSB.
//   AW/W/B: narrow engine only.
// =============================================================================
module hbm_port_mux import llm_pkg::*; (
  input  logic clk, rst_n,
  input  logic err_clr,
  output logic axi_err,
  input  logic [N_HBM-1:0][63:0] hbm_base,

  input  logic [N_HBM-1:0][AXI_AW-1:0]   ws_araddr,
  input  logic [N_HBM-1:0][7:0]          ws_arlen,
  input  logic [N_HBM-1:0]               ws_arvalid,
  output logic [N_HBM-1:0]               ws_arready,
  output logic [N_HBM-1:0][AXI_DW-1:0]   ws_rdata,
  output logic [N_HBM-1:0]               ws_rvalid, ws_rlast,
  input  logic [N_HBM-1:0]               ws_rready,

  input  logic [N_HBM-1:0][AXI_AW-1:0]   nr_araddr,
  input  logic [N_HBM-1:0][7:0]          nr_arlen,
  input  logic [N_HBM-1:0]               nr_arvalid,
  output logic [N_HBM-1:0]               nr_arready,
  output logic [N_HBM-1:0][AXI_DW-1:0]   nr_rdata,
  output logic [N_HBM-1:0]               nr_rvalid, nr_rlast,
  input  logic [N_HBM-1:0]               nr_rready,
  input  logic [N_HBM-1:0][AXI_AW-1:0]   nr_awaddr,
  input  logic [N_HBM-1:0][7:0]          nr_awlen,
  input  logic [N_HBM-1:0]               nr_awvalid,
  output logic [N_HBM-1:0]               nr_awready,
  input  logic [N_HBM-1:0][AXI_DW-1:0]   nr_wdata,
  input  logic [N_HBM-1:0][AXI_DW/8-1:0] nr_wstrb,
  input  logic [N_HBM-1:0]               nr_wlast,
  input  logic [N_HBM-1:0]               nr_wvalid,
  output logic [N_HBM-1:0]               nr_wready,
  output logic [N_HBM-1:0]               nr_bvalid,
  input  logic [N_HBM-1:0]               nr_bready,

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
  localparam logic [AXI_IDW-1:0] ID_WS = '0;
  localparam logic [AXI_IDW-1:0] ID_NR = {1'b1, {(AXI_IDW-1){1'b0}}};

  logic [N_HBM-1:0] err_now;

  for (genvar p = 0; p < N_HBM; p++) begin : g_port
    logic lock_q, grant_q, g, is_nr;

    assign g = lock_q ? grant_q : nr_arvalid[p];          // 1 = narrow

    // Port p only ever addresses its own PC: drop {pc} and add the host buffer base.
    assign m_axi_araddr[p]  = M_AXI_AW'(hbm_base[p]) + M_AXI_AW'((g ? nr_araddr[p] : ws_araddr[p]) & AXI_AW'(29'h1FFF_FFFF));
    assign m_axi_arlen[p]   = g ? nr_arlen[p]   : ws_arlen[p];
    assign m_axi_arid[p]    = g ? ID_NR         : ID_WS;
    assign m_axi_arvalid[p] = g ? nr_arvalid[p] : ws_arvalid[p];
    assign nr_arready[p]    =  g && m_axi_arready[p];
    assign ws_arready[p]    = !g && m_axi_arready[p];

    always_ff @(posedge clk) begin
      if (!rst_n) begin
        lock_q <= 1'b0; grant_q <= 1'b0;
      end else begin
        lock_q  <= m_axi_arvalid[p] && !m_axi_arready[p];
        grant_q <= g;
      end
    end

    assign is_nr           = m_axi_rid[p][AXI_IDW-1];
    assign ws_rdata[p]     = m_axi_rdata[p];
    assign nr_rdata[p]     = m_axi_rdata[p];
    assign ws_rlast[p]     = m_axi_rlast[p];
    assign nr_rlast[p]     = m_axi_rlast[p];
    assign ws_rvalid[p]    = m_axi_rvalid[p] && !is_nr;
    assign nr_rvalid[p]    = m_axi_rvalid[p] &&  is_nr;
    assign m_axi_rready[p] = is_nr ? nr_rready[p] : ws_rready[p];

    assign m_axi_awaddr[p]  = M_AXI_AW'(hbm_base[p]) + M_AXI_AW'(nr_awaddr[p] & AXI_AW'(29'h1FFF_FFFF));
    assign m_axi_awlen[p]   = nr_awlen[p];
    assign m_axi_awid[p]    = ID_NR;
    assign m_axi_awvalid[p] = nr_awvalid[p];
    assign nr_awready[p]    = m_axi_awready[p];
    assign m_axi_wdata[p]   = nr_wdata[p];
    assign m_axi_wstrb[p]   = nr_wstrb[p];
    assign m_axi_wlast[p]   = nr_wlast[p];
    assign m_axi_wvalid[p]  = nr_wvalid[p];
    assign nr_wready[p]     = m_axi_wready[p];
    assign nr_bvalid[p]     = m_axi_bvalid[p];
    assign m_axi_bready[p]  = nr_bready[p];

    assign err_now[p] = (m_axi_rvalid[p] && m_axi_rready[p] && (m_axi_rresp[p] != 2'b00)) ||
                        (m_axi_bvalid[p] && m_axi_bready[p] && (m_axi_bresp[p] != 2'b00));
  end

  always_ff @(posedge clk) begin
    if (!rst_n || err_clr) axi_err <= 1'b0;
    else if (|err_now)     axi_err <= 1'b1;
  end

  logic unused_bid;
  assign unused_bid = ^m_axi_bid;
endmodule


// =============================================================================
// narrow_hbm_engine
//   Read : one request at a time. Bursts <= 64 beats, never cross 4 KB. Up to 8 bursts in flight,
//          all on one PC; switching PC waits for drain (keeps data in order without reordering).
//   Write: router requests (full-beat, burst) OR kv ops (single beat, byte strobes), never both
//          at once. Up to 15 bursts queued for W; B responses counted. wdone after all B's.
// =============================================================================
module narrow_hbm_engine import llm_pkg::*; (
  input  logic clk, rst_n,

  // read server (from addr_router)
  input  mem_req_t                       rq,
  input  logic                           rq_v,
  output logic                           rq_r,
  output logic [BEAT_W-1:0]              rdata,
  output logic                           rv, rl,
  input  logic                           rr,

  // write server (from addr_router)
  input  mem_req_t                       wq,
  input  logic                           wq_v,
  output logic                           wq_r,
  input  logic [BEAT_W-1:0]              wdata,
  input  logic                           wv, wl,
  output logic                           wr,
  output logic                           wdone,

  // kv op port (from kv_cache_mgr): one beat per op
  input  logic                           kv_hold,
  input  logic [AXI_AW-1:0]              kv_addr,
  input  logic [AXI_DW-1:0]              kv_data,
  input  logic [AXI_DW/8-1:0]            kv_strb,
  input  logic                           kv_v,
  output logic                           kv_r,
  output logic                           w_quiet,

  // per-port AXI (to hbm_port_mux)
  output logic [N_HBM-1:0][AXI_AW-1:0]   nr_araddr,
  output logic [N_HBM-1:0][7:0]          nr_arlen,
  output logic [N_HBM-1:0]               nr_arvalid,
  input  logic [N_HBM-1:0]               nr_arready,
  input  logic [N_HBM-1:0][AXI_DW-1:0]   nr_rdata,
  input  logic [N_HBM-1:0]               nr_rvalid, nr_rlast,
  output logic [N_HBM-1:0]               nr_rready,
  output logic [N_HBM-1:0][AXI_AW-1:0]   nr_awaddr,
  output logic [N_HBM-1:0][7:0]          nr_awlen,
  output logic [N_HBM-1:0]               nr_awvalid,
  input  logic [N_HBM-1:0]               nr_awready,
  output logic [N_HBM-1:0][AXI_DW-1:0]   nr_wdata,
  output logic [N_HBM-1:0][AXI_DW/8-1:0] nr_wstrb,
  output logic [N_HBM-1:0]               nr_wlast,
  output logic [N_HBM-1:0]               nr_wvalid,
  input  logic [N_HBM-1:0]               nr_wready,
  input  logic [N_HBM-1:0]               nr_bvalid,
  output logic [N_HBM-1:0]               nr_bready
);
  localparam int NB = 64;    // max burst beats

  // burst length for a 2D walk: min(NB, beats left in row, beats to 4 KB boundary)
  function automatic logic [15:0] blen(input logic [AXI_AW-1:0] a, input logic [15:0] left);
    logic [15:0] l, to4k;
    to4k = 16'd128 - {9'd0, a[11:5]};
    l = 16'(NB);
    if (left < l) l = left;
    if (to4k < l) l = to4k;
    return l;
  endfunction

  // ===========================================================================
  // READ
  // ===========================================================================
  logic              r_busy, r_ar_done;
  logic [AXI_AW-1:0] r_row_addr, r_a;
  logic [15:0]       r_col, r_row, r_rbeats, r_rows, r_len;
  logic [31:0]       r_stride, r_total, r_cnt;
  logic [4:0]        r_pc, r_pc_a;
  logic [3:0]        r_out;              // bursts in flight on r_pc
  logic              r_can, r_ar_hs, r_hs, r_burst_end;

  assign r_a    = r_row_addr + AXI_AW'({r_col, 5'd0});
  assign r_len  = blen(r_a, r_rbeats - r_col);
  assign r_pc_a = r_a[AXI_AW-1 -: 5];
  assign r_can  = r_busy && !r_ar_done && (r_out == 0 || (r_pc_a == r_pc && r_out < 4'd8));
  assign r_ar_hs = r_can && nr_arready[r_pc_a];
  assign rq_r   = !r_busy;

  for (genvar p = 0; p < N_HBM; p++) begin : g_rd
    assign nr_araddr[p]  = r_a;
    assign nr_arlen[p]   = 8'(r_len - 16'd1);
    assign nr_arvalid[p] = r_can && (r_pc_a == 5'(p));
    assign nr_rready[p]  = r_busy && (r_pc == 5'(p)) && rr;
  end

  assign rdata       = nr_rdata[r_pc];
  assign rv          = r_busy && (r_out != 0) && nr_rvalid[r_pc];
  assign rl          = (r_cnt == r_total - 1);
  assign r_hs        = rv && rr;
  assign r_burst_end = r_hs && nr_rlast[r_pc];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      r_busy <= 1'b0; r_ar_done <= 1'b1; r_out <= '0; r_pc <= '0;
      r_row_addr <= '0; r_col <= '0; r_row <= '0; r_rbeats <= '0; r_rows <= '0;
      r_stride <= '0; r_total <= '0; r_cnt <= '0;
    end else begin
      if (rq_v && rq_r) begin
        r_busy     <= (rq.row_beats != 0) && (rq.n_rows != 0);
        r_ar_done  <= 1'b0;
        r_row_addr <= rq.base.off;
        r_col      <= '0;
        r_row      <= '0;
        r_rbeats   <= rq.row_beats;
        r_rows     <= rq.n_rows;
        r_stride   <= rq.row_stride;
        r_total    <= 32'(rq.row_beats) * 32'(rq.n_rows);
        r_cnt      <= '0;
      end else begin
        if (r_ar_hs) begin
          r_pc <= r_pc_a;
          if (r_col + r_len == r_rbeats) begin
            r_col      <= '0;
            r_row      <= r_row + 1;
            r_row_addr <= r_row_addr + AXI_AW'(r_stride);
            if (r_row + 1 == r_rows) r_ar_done <= 1'b1;
          end else begin
            r_col <= r_col + r_len;
          end
        end
        if (r_hs) begin
          r_cnt <= r_cnt + 1;
          if (rl) r_busy <= 1'b0;
        end
      end
      r_out <= r_out + (r_ar_hs ? 4'd1 : 4'd0) - (r_burst_end ? 4'd1 : 4'd0);
    end
  end

  // ===========================================================================
  // WRITE
  // ===========================================================================
  logic              w_busy, w_aw_done, w_fin;
  logic [AXI_AW-1:0] w_row_addr, w_a, aw_addr;
  logic [15:0]       w_col, w_row, w_rbeats, w_rows, w_len, aw_len;
  logic [31:0]       w_stride, w_total, w_cnt;
  logic [4:0]        aw_pc;
  logic              aw_v, aw_hs, aw_kv;
  logic [7:0]        outst;              // AW issued, B not yet received
  logic [5:0]        b_cnt;

  // route FIFO: which port / how many beats / source, for each issued AW
  logic [13:0] rt_din, rt_dout;
  logic        rt_full, rt_valid, rt_pop;
  logic [5:0]  rt_count;
  logic [4:0]  rt_pc;
  logic [7:0]  rt_len;
  logic        rt_kv;
  logic [7:0]  w_beat;

  // kv data FIFO
  logic [AXI_DW+AXI_DW/8-1:0] kd_dout;
  logic        kd_full, kd_valid, kd_pop;

  assign w_a    = w_row_addr + AXI_AW'({w_col, 5'd0});
  assign w_len  = blen(w_a, w_rbeats - w_col);
  assign wq_r   = !w_busy && !kv_hold;

  // AW source: router burst while w_busy, else kv op while kv_hold
  assign aw_kv   = !w_busy;
  assign aw_addr = w_busy ? w_a   : kv_addr;
  assign aw_len  = w_busy ? w_len : 16'd1;
  assign aw_pc   = aw_addr[AXI_AW-1 -: 5];
  assign aw_v    = !rt_full && (w_busy ? !w_aw_done : (kv_hold && kv_v && !kd_full));
  assign aw_hs   = aw_v && nr_awready[aw_pc];
  assign kv_r    = aw_hs && aw_kv;

  for (genvar p = 0; p < N_HBM; p++) begin : g_aw
    assign nr_awaddr[p]  = aw_addr;
    assign nr_awlen[p]   = 8'(aw_len - 16'd1);
    assign nr_awvalid[p] = aw_v && (aw_pc == 5'(p));
  end

  assign rt_din = {aw_pc, 8'(aw_len), aw_kv};
  t4_fifo #(.W(14), .DEPTH(16)) u_rt (
    .clk, .rst_n, .din (rt_din), .push (aw_hs), .full (rt_full),
    .dout (rt_dout), .valid (rt_valid), .pop (rt_pop), .count (rt_count)
  );
  assign {rt_pc, rt_len, rt_kv} = rt_dout;

  t4_fifo #(.W(AXI_DW + AXI_DW/8), .DEPTH(16)) u_kd (
    .clk, .rst_n, .din ({kv_data, kv_strb}), .push (kv_r), .full (kd_full),
    .dout (kd_dout), .valid (kd_valid), .pop (kd_pop), .count ()
  );

  // W channel: follow the route FIFO head
  logic                src_v, w_hs, w_last;
  logic [AXI_DW-1:0]   src_d;
  logic [AXI_DW/8-1:0] src_s;

  assign src_v  = rt_kv ? kd_valid : (w_busy && wv);
  assign src_d  = rt_kv ? kd_dout[AXI_DW+AXI_DW/8-1 -: AXI_DW] : wdata;
  assign src_s  = rt_kv ? kd_dout[AXI_DW/8-1:0] : '1;
  assign w_last = (w_beat == rt_len - 8'd1);
  assign w_hs   = rt_valid && src_v && nr_wready[rt_pc];
  assign rt_pop = w_hs && w_last;
  assign kd_pop = w_hs && rt_kv;
  assign wr     = rt_valid && !rt_kv && w_busy && nr_wready[rt_pc];

  for (genvar p = 0; p < N_HBM; p++) begin : g_w
    assign nr_wdata[p]  = src_d;
    assign nr_wstrb[p]  = src_s;
    assign nr_wlast[p]  = w_last;
    assign nr_wvalid[p] = rt_valid && src_v && (rt_pc == 5'(p));
    assign nr_bready[p] = 1'b1;
  end

  always_comb begin
    b_cnt = '0;
    for (int p = 0; p < N_HBM; p++) b_cnt = b_cnt + 6'(nr_bvalid[p]);
  end

  assign w_quiet = (outst == 0) && !rt_valid && (rt_count == 0);
  assign w_fin   = w_busy && w_aw_done && (w_cnt == w_total) && w_quiet && !aw_hs;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      w_busy <= 1'b0; w_aw_done <= 1'b1; w_beat <= '0; outst <= '0; wdone <= 1'b0;
      w_row_addr <= '0; w_col <= '0; w_row <= '0; w_rbeats <= '0; w_rows <= '0;
      w_stride <= '0; w_total <= '0; w_cnt <= '0;
    end else begin
      wdone <= 1'b0;
      if (wq_v && wq_r) begin
        w_busy     <= 1'b1;
        w_aw_done  <= (wq.row_beats == 0) || (wq.n_rows == 0);
        w_row_addr <= wq.base.off;
        w_col      <= '0;
        w_row      <= '0;
        w_rbeats   <= wq.row_beats;
        w_rows     <= wq.n_rows;
        w_stride   <= wq.row_stride;
        w_total    <= 32'(wq.row_beats) * 32'(wq.n_rows);
        w_cnt      <= '0;
      end else if (w_fin) begin
        w_busy <= 1'b0;
        wdone  <= 1'b1;
      end else if (aw_hs && !aw_kv) begin
        if (w_col + w_len == w_rbeats) begin
          w_col      <= '0;
          w_row      <= w_row + 1;
          w_row_addr <= w_row_addr + AXI_AW'(w_stride);
          if (w_row + 1 == w_rows) w_aw_done <= 1'b1;
        end else begin
          w_col <= w_col + w_len;
        end
      end
      if (w_hs) begin
        w_beat <= w_last ? '0 : w_beat + 1;
        if (!rt_kv) w_cnt <= w_cnt + 1;
      end
      outst <= outst + (aw_hs ? 8'd1 : 8'd0) - 8'(b_cnt);
    end
  end

  logic unused_wl;
  assign unused_wl = wl;
endmodule


// =============================================================================
// addr_router: NRD read clients, NWR write clients -> 2 servers (0 = act_buf, 1 = HBM).
//   A server is owned by one client from req handshake until its last read beat / write done.
//   A client owned by one server is not granted the other (keeps its data port unambiguous).
//   Arbitration: lowest client index wins (fixed priority).
// =============================================================================
module addr_router import llm_pkg::*; #(
  parameter int NRD = 6,
  parameter int NWR = 4
)(
  input  logic clk, rst_n,

  input  mem_req_t [NRD-1:0]             c_rq,
  input  logic     [NRD-1:0]             c_rq_v,
  output logic     [NRD-1:0]             c_rq_r,
  output logic     [NRD-1:0][BEAT_W-1:0] c_rd,
  output logic     [NRD-1:0]             c_rv, c_rl,
  input  logic     [NRD-1:0]             c_rr,

  input  mem_req_t [NWR-1:0]             c_wq,
  input  logic     [NWR-1:0]             c_wq_v,
  output logic     [NWR-1:0]             c_wq_r,
  input  logic     [NWR-1:0][BEAT_W-1:0] c_wd,
  input  logic     [NWR-1:0]             c_wv, c_wl,
  output logic     [NWR-1:0]             c_wr, c_wdone,

  output mem_req_t [1:0]                 s_rq,
  output logic     [1:0]                 s_rq_v,
  input  logic     [1:0]                 s_rq_r,
  input  logic     [1:0][BEAT_W-1:0]     s_rd,
  input  logic     [1:0]                 s_rv, s_rl,
  output logic     [1:0]                 s_rr,

  output mem_req_t [1:0]                 s_wq,
  output logic     [1:0]                 s_wq_v,
  input  logic     [1:0]                 s_wq_r,
  output logic     [1:0][BEAT_W-1:0]     s_wd,
  output logic     [1:0]                 s_wv, s_wl,
  input  logic     [1:0]                 s_wr, s_wdone
);
  localparam int RW = (NRD > 1) ? $clog2(NRD) : 1;
  localparam int WW = (NWR > 1) ? $clog2(NWR) : 1;

  function automatic logic tgt(input addr_t a);
    return (a.rgn == RGN_HBM);
  endfunction

  // ===================== READ =====================
  logic [1:0]          r_own_v, r_cand_v;
  logic [1:0][RW-1:0]  r_own, r_cand;
  logic [NRD-1:0]      r_busy;

  always_comb begin
    r_busy = '0;
    for (int s = 0; s < 2; s++) if (r_own_v[s]) r_busy[r_own[s]] = 1'b1;
    for (int s = 0; s < 2; s++) begin
      r_cand_v[s] = 1'b0;
      r_cand[s]   = '0;
      for (int i = NRD-1; i >= 0; i--)
        if (c_rq_v[i] && !r_busy[i] && (tgt(c_rq[i].base) == s[0])) begin
          r_cand_v[s] = 1'b1;
          r_cand[s]   = RW'(i);
        end
      s_rq[s]   = c_rq[r_cand[s]];
      s_rq_v[s] = !r_own_v[s] && r_cand_v[s];
      s_rr[s]   = r_own_v[s] && c_rr[r_own[s]];
    end
    for (int i = 0; i < NRD; i++) begin
      c_rq_r[i] = 1'b0;
      c_rv[i]   = 1'b0;
      c_rl[i]   = 1'b0;
      c_rd[i]   = s_rd[0];
      for (int s = 0; s < 2; s++) begin
        if (s_rq_v[s] && r_cand[s] == RW'(i) && s_rq_r[s]) c_rq_r[i] = 1'b1;
        if (r_own_v[s] && r_own[s] == RW'(i)) begin
          c_rv[i] = s_rv[s];
          c_rl[i] = s_rl[s];
          c_rd[i] = s_rd[s];
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      r_own_v <= '0; r_own <= '0;
    end else begin
      for (int s = 0; s < 2; s++) begin
        if (s_rq_v[s] && s_rq_r[s]) begin
          r_own_v[s] <= 1'b1;
          r_own[s]   <= r_cand[s];
        end else if (r_own_v[s] && s_rv[s] && s_rr[s] && s_rl[s]) begin
          r_own_v[s] <= 1'b0;
        end
      end
    end
  end

  // ===================== WRITE =====================
  logic [1:0]          w_own_v, w_cand_v;
  logic [1:0][WW-1:0]  w_own, w_cand;
  logic [NWR-1:0]      w_busy;

  always_comb begin
    w_busy = '0;
    for (int s = 0; s < 2; s++) if (w_own_v[s]) w_busy[w_own[s]] = 1'b1;
    for (int s = 0; s < 2; s++) begin
      w_cand_v[s] = 1'b0;
      w_cand[s]   = '0;
      for (int i = NWR-1; i >= 0; i--)
        if (c_wq_v[i] && !w_busy[i] && (tgt(c_wq[i].base) == s[0])) begin
          w_cand_v[s] = 1'b1;
          w_cand[s]   = WW'(i);
        end
      s_wq[s]   = c_wq[w_cand[s]];
      s_wq_v[s] = !w_own_v[s] && w_cand_v[s];
      s_wd[s]   = c_wd[w_own[s]];
      s_wv[s]   = w_own_v[s] && c_wv[w_own[s]];
      s_wl[s]   = c_wl[w_own[s]];
    end
    for (int i = 0; i < NWR; i++) begin
      c_wq_r[i]  = 1'b0;
      c_wr[i]    = 1'b0;
      c_wdone[i] = 1'b0;
      for (int s = 0; s < 2; s++) begin
        if (s_wq_v[s] && w_cand[s] == WW'(i) && s_wq_r[s]) c_wq_r[i] = 1'b1;
        if (w_own_v[s] && w_own[s] == WW'(i)) begin
          c_wr[i]    = s_wr[s];
          c_wdone[i] = s_wdone[s];
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      w_own_v <= '0; w_own <= '0;
    end else begin
      for (int s = 0; s < 2; s++) begin
        if (s_wq_v[s] && s_wq_r[s]) begin
          w_own_v[s] <= 1'b1;
          w_own[s]   <= w_cand[s];
        end else if (w_own_v[s] && s_wdone[s]) begin
          w_own_v[s] <= 1'b0;
        end
      end
    end
  end
endmodule


// =============================================================================
// dma_engine: copy len_beats from src to dst (any region). One copy at a time.
//   len_beats must be < 65536 (one row). OP_EMBED = 128 beats HBM -> ACT.
// =============================================================================
module dma_engine import llm_pkg::*; (
  input  logic clk, rst_n,
  input  dma_cmd_t           cmd,
  input  logic               cmd_v,
  output logic               cmd_r,
  output logic               done,

  output mem_req_t           rq,
  output logic               rq_v,
  input  logic               rq_r,
  input  logic [BEAT_W-1:0]  rd,
  input  logic               rv, rl,
  output logic               rr,

  output mem_req_t           wq,
  output logic               wq_v,
  input  logic               wq_r,
  output logic [BEAT_W-1:0]  wd,
  output logic               wv, wl,
  input  logic               wr,
  input  logic               wdone
);
  logic     busy, rq_sent, wq_sent, both;
  dma_cmd_t c;

  assign cmd_r = !busy;
  assign both  = rq_sent && wq_sent;

  assign rq = '{base: c.src, row_beats: c.len_beats[15:0], n_rows: 16'd1, row_stride: 32'd0};
  assign wq = '{base: c.dst, row_beats: c.len_beats[15:0], n_rows: 16'd1, row_stride: 32'd0};
  assign rq_v = busy && !rq_sent;
  assign wq_v = busy && !wq_sent;

  assign wd = rd;
  assign wl = rl;
  assign wv = both && rv;
  assign rr = both && wr;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      busy <= 1'b0; rq_sent <= 1'b0; wq_sent <= 1'b0; done <= 1'b0; c <= '0;
    end else begin
      done <= 1'b0;
      if (cmd_v && cmd_r) begin
        busy <= 1'b1; rq_sent <= 1'b0; wq_sent <= 1'b0; c <= cmd;
      end else if (busy) begin
        if (rq_v && rq_r) rq_sent <= 1'b1;
        if (wq_v && wq_r) wq_sent <= 1'b1;
        if (wdone) begin
          busy <= 1'b0; done <= 1'b1;
        end
      end
    end
  end
endmodule


// =============================================================================
// act_buf: on-chip activations (RGN_ACT) + scratch (RGN_SCRATCH) in one URAM array.
//   ACT     = beats [0, ACT_BEATS)                 128 KB default
//   SCRATCH = beats [ACT_BEATS, ACT_BEATS+SCR_BEATS) 384 KB default (logits = 251 KB)
//   1 read port (2-cycle latency, 1 beat/cycle sustained) + 1 write port, independent.
//   wdone pulses the cycle after the last beat is written.
// =============================================================================
module act_buf import llm_pkg::*; #(
  parameter int ACT_BEATS = 4096,
  parameter int SCR_BEATS = 12288
)(
  input  logic clk, rst_n,

  input  mem_req_t           rq,
  input  logic               rq_v,
  output logic               rq_r,
  output logic [BEAT_W-1:0]  rdata,
  output logic               rv, rl,
  input  logic               rr,

  input  mem_req_t           wq,
  input  logic               wq_v,
  output logic               wq_r,
  input  logic [BEAT_W-1:0]  wdata,
  input  logic               wv, wl,
  output logic               wr,
  output logic               wdone
);
  localparam int DEPTH = ACT_BEATS + SCR_BEATS;
  localparam int AW    = $clog2(DEPTH);

  function automatic logic [AW-1:0] beat_of(input addr_t a);
    logic [AW-1:0] b;
    b = AW'(a.off >> 5);
    return (a.rgn == RGN_SCRATCH) ? AW'(ACT_BEATS) + b : b;
  endfunction

  (* ram_style = "ultra" *) logic [BEAT_W-1:0] mem [DEPTH];

  // ===================== READ =====================
  logic          r_busy, r_iss_done, issue, last_iss;
  logic [AW-1:0] r_base, r_addr, r_stride;
  logic [15:0]   r_col, r_row, r_rbeats, r_rows;
  logic          s1_v, s1_l, s2_v, s2_l;
  logic [BEAT_W-1:0] s1_d, s2_d;
  logic [BEAT_W:0]   of_dout;
  logic          of_valid, of_pop;
  logic [3:0]    of_count;

  assign rq_r     = !r_busy;
  assign r_addr   = r_base + AW'(r_col);
  assign last_iss = (r_col == r_rbeats - 1) && (r_row == r_rows - 1);
  assign issue    = r_busy && !r_iss_done &&
                    ({1'b0, of_count} + 5'(s1_v) + 5'(s2_v) < 5'd4);

  always_ff @(posedge clk) begin
    if (issue) s1_d <= mem[r_addr];
    s2_d <= s1_d;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      r_busy <= 1'b0; r_iss_done <= 1'b1; s1_v <= 1'b0; s2_v <= 1'b0; s1_l <= 1'b0; s2_l <= 1'b0;
      r_base <= '0; r_stride <= '0; r_col <= '0; r_row <= '0; r_rbeats <= '0; r_rows <= '0;
    end else begin
      s1_v <= issue;  s1_l <= issue && last_iss;
      s2_v <= s1_v;   s2_l <= s1_l;
      if (rq_v && rq_r) begin
        r_busy     <= (rq.row_beats != 0) && (rq.n_rows != 0);
        r_iss_done <= 1'b0;
        r_base     <= beat_of(rq.base);
        r_stride   <= AW'(rq.row_stride >> 5);
        r_col      <= '0;
        r_row      <= '0;
        r_rbeats   <= rq.row_beats;
        r_rows     <= rq.n_rows;
      end else begin
        if (issue) begin
          if (r_col == r_rbeats - 1) begin
            r_col  <= '0;
            r_row  <= r_row + 1;
            r_base <= r_base + r_stride;
            if (last_iss) r_iss_done <= 1'b1;
          end else begin
            r_col <= r_col + 1;
          end
        end
        if (of_pop && rl) r_busy <= 1'b0;
      end
    end
  end

  t4_fifo #(.W(BEAT_W + 1), .DEPTH(4)) u_of (
    .clk, .rst_n, .din ({s2_l, s2_d}), .push (s2_v), .full (),
    .dout (of_dout), .valid (of_valid), .pop (of_pop), .count (of_count)
  );
  assign rdata  = of_dout[BEAT_W-1:0];
  assign rl     = of_dout[BEAT_W];
  assign rv     = of_valid;
  assign of_pop = rv && rr;

  // ===================== WRITE =====================
  logic          w_busy, w_last;
  logic [AW-1:0] w_base, w_stride;
  logic [15:0]   w_col, w_row, w_rbeats, w_rows;

  assign wq_r   = !w_busy;
  assign wr     = w_busy;
  assign w_last = (w_col == w_rbeats - 1) && (w_row == w_rows - 1);

  always_ff @(posedge clk) begin
    if (w_busy && wv) mem[w_base + AW'(w_col)] <= wdata;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      w_busy <= 1'b0; wdone <= 1'b0;
      w_base <= '0; w_stride <= '0; w_col <= '0; w_row <= '0; w_rbeats <= '0; w_rows <= '0;
    end else begin
      wdone <= 1'b0;
      if (wq_v && wq_r) begin
        w_busy   <= 1'b1;
        w_base   <= beat_of(wq.base);
        w_stride <= AW'(wq.row_stride >> 5);
        w_col    <= '0;
        w_row    <= '0;
        w_rbeats <= wq.row_beats;
        w_rows   <= wq.n_rows;
        if (wq.row_beats == 0 || wq.n_rows == 0) begin w_busy <= 1'b0; wdone <= 1'b1; end
      end else if (w_busy && wv) begin
        if (w_last) begin
          w_busy <= 1'b0;
          wdone  <= 1'b1;
        end else if (w_col == w_rbeats - 1) begin
          w_col  <= '0;
          w_row  <= w_row + 1;
          w_base <= w_base + w_stride;
        end else begin
          w_col <= w_col + 1;
        end
      end
    end
  end

  logic unused_wl;
  assign unused_wl = wl;
endmodule


// =============================================================================
// kv_cache_mgr: one kv_wr request = one head's K or V vector (HEAD_DIM int8 = 2 beats) at pos.
//   Tag in kv.req.base.off[28:0] (kv_tag_t). Address math per llm_pkg KV CONTRACT.
//   V: 2 full-beat writes to PC (pos % 32).
//   K: K^T column write -> HEAD_DIM single-byte writes (PC = dim % 32, byte = pos % 32).
//      Baseline cost: 64 small writes per head, 512 per layer. Pipelined (B not awaited per op).
//      Later: buffer 32 positions on-chip and flush full beats.
//   done pulses after every write's B response is back (data visible to the streamer).
// =============================================================================
module kv_cache_mgr import llm_pkg::*; (
  input  logic clk, rst_n,
  mem_wr_if.server kv,

  output logic                kv_hold,
  output logic [AXI_AW-1:0]   op_addr,
  output logic [AXI_DW-1:0]   op_data,
  output logic [AXI_DW/8-1:0] op_strb,
  output logic                op_v,
  input  logic                op_r,
  input  logic                w_quiet
);
  localparam int VB   = HEAD_DIM / 32;      // beats per vector (2)
  localparam int KT_K = HEAD_DIM / 32;      // k_tiles of K^T (2)
  localparam int KT_V = MAX_SEQ / 32;       // k_tiles of V (64)

  typedef enum logic [1:0] {S_IDLE, S_DATA, S_OPS, S_DRAIN} st_e;
  st_e st;

  kv_tag_t               tag;
  logic [VB-1:0][AXI_DW-1:0] vbuf;
  logic [$clog2(VB+1)-1:0]   beat;
  logic [7:0]            i;               // op index
  logic [7:0]            n_ops;
  logic [28:0]           blk;

  assign kv.req_ready = (st == S_IDLE);
  assign kv.ready     = (st == S_DATA);
  assign kv_hold      = (st == S_OPS) || (st == S_DRAIN);
  assign op_v         = (st == S_OPS);
  assign n_ops        = tag.is_v ? 8'(VB) : 8'(HEAD_DIM);
  assign blk          = kv_pc_off(tag.layer, tag.kvh, tag.is_v);

  always_comb begin
    logic [4:0]  pc;
    logic [28:0] beat_idx;
    logic [7:0]  val;
    val = '0;
    if (tag.is_v) begin
      // V tile row = position; n_tile = i (dims i*32 .. i*32+31)
      pc       = tag.pos[4:0];
      beat_idx = 29'(i) * 29'(KT_V) + 29'(tag.pos[15:5]);
      op_data  = vbuf[i[$clog2(VB)-1:0]];
      op_strb  = '1;
    end else begin
      // K^T tile row = dim i; n_tile = pos/32; column = pos%32
      pc       = i[4:0];
      beat_idx = 29'(tag.pos[15:5]) * 29'(KT_K) + 29'(i[7:5]);
      val      = vbuf[i[7:5]][i[4:0]*8 +: 8];
      op_data  = {(AXI_DW/8){val}};
      op_strb  = (AXI_DW/8)'(1) << tag.pos[4:0];
    end
    op_addr = {pc, blk + (beat_idx << 5)};
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      st <= S_IDLE; tag <= '0; beat <= '0; i <= '0; kv.done <= 1'b0; vbuf <= '0;
    end else begin
      kv.done <= 1'b0;
      case (st)
        S_IDLE: if (kv.req_valid) begin
          tag  <= kv_tag_t'(kv.req.base.off[28:0]);
          beat <= '0;
          st   <= S_DATA;
        end
        S_DATA: if (kv.valid) begin
          vbuf[beat[$clog2(VB)-1:0]] <= kv.data[AXI_DW-1:0];
          beat <= beat + 1;
          if (32'(beat) == VB - 1) begin i <= '0; st <= S_OPS; end
        end
        S_OPS: if (op_r) begin
          i <= i + 1;
          if (i == n_ops - 1) st <= S_DRAIN;
        end
        S_DRAIN: if (w_quiet) begin
          kv.done <= 1'b1;
          st      <= S_IDLE;
        end
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule


// =============================================================================
// ctrl_regs: Vitis ap_ctrl_hs AXI-Lite slave. Offsets MUST match kernel.xml.
//   0x00 ap_ctrl : [0] ap_start (RW, clears on done) [1] ap_done (COR) [2] ap_idle (RO)
//                  [3] ap_ready (COR)
//   0x04 GIE [0]   0x08 IER [1:0]   0x0C ISR [1:0] (toggle-on-write)
//   0x10 instr_base[31:0]  0x14 instr_base[63:32] (low 34 bits used, region = HBM)
//   0x18 n_instr   0x1C token_pos   0x20 result_token (RO)   0x24 token_in
//   RO perf (cleared on every start):
//   0x40/0x44 kernel cycles   0x48/0x4C ws_beats   0x50/0x54 ws_active
//   0x58/0x5C ws_starve       0x60/0x64 ws_stall   0x68 [0] axi_err
//   0x100 + 8*p (lo) / +4 (hi): HBM buffer base for m{p}_axi (host pointer arg), reset = p*512MB
//   ap_start (port) = 1-cycle pulse to llm_ctrl when host writes start while idle.
// =============================================================================
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
  output logic [31:0] token_in,
  input  logic [31:0] result_token,
  input  perf_t       perf,
  output logic        perf_clr,
  output logic [N_HBM-1:0][63:0] hbm_base
);
  logic [N_HBM-1:0][63:0] base_q;
  assign hbm_base = base_q;
  logic        aw_have, w_have, busy, start_bit, done_bit, ready_bit, gie;
  logic [11:0] aw_q;
  logic [31:0] w_q, base_lo, base_hi, n_instr_q, pos_q, tok_q;
  logic [1:0]  ier, isr;
  logic [63:0] cycles;
  logic        do_wr, do_rd;

  // byte-enable merge
  function automatic logic [31:0] merge(input logic [31:0] old, input logic [31:0] d,
                                        input logic [3:0] be);
    for (int b = 0; b < 4; b++) if (be[b]) old[b*8 +: 8] = d[b*8 +: 8];
    return old;
  endfunction
  logic [3:0] wstrb_q;

  assign awready = !aw_have;
  assign wready  = !w_have;
  assign bresp   = 2'b00;
  assign rresp   = 2'b00;
  assign arready = !rvalid;
  assign do_wr   = aw_have && w_have && !bvalid;
  assign do_rd   = arvalid && arready;

  assign instr_base = '{rgn: RGN_HBM, off: {base_hi[1:0], base_lo}};
  assign n_instr    = n_instr_q[15:0];
  assign token_pos  = pos_q[15:0];
  assign token_in   = tok_q;
  assign interrupt  = gie && |(isr & ier);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      aw_have <= 1'b0; w_have <= 1'b0; bvalid <= 1'b0; rvalid <= 1'b0; rdata <= '0;
      aw_q <= '0; w_q <= '0; wstrb_q <= '0;
      busy <= 1'b0; start_bit <= 1'b0; done_bit <= 1'b0; ready_bit <= 1'b0;
      gie <= 1'b0; ier <= '0; isr <= '0;
      base_lo <= '0; base_hi <= '0; n_instr_q <= '0; pos_q <= '0; tok_q <= '0;
      ap_start <= 1'b0; perf_clr <= 1'b0; cycles <= '0;
      for (int p = 0; p < N_HBM; p++) base_q[p] <= 64'(p) << 29;
    end else begin
      ap_start <= 1'b0;
      perf_clr <= 1'b0;

      // ---- write channel
      if (awvalid && awready) begin aw_have <= 1'b1; aw_q <= awaddr; end
      if (wvalid && wready)   begin w_have <= 1'b1; w_q <= wdata; wstrb_q <= wstrb; end
      if (bvalid && bready)   bvalid <= 1'b0;
      if (do_wr) begin
        aw_have <= 1'b0; w_have <= 1'b0; bvalid <= 1'b1;
        case (aw_q)
          12'h000: if (wstrb_q[0] && w_q[0] && !busy) begin
                     start_bit <= 1'b1; busy <= 1'b1; ap_start <= 1'b1; perf_clr <= 1'b1;
                     cycles <= '0;
                   end
          12'h004: gie       <= w_q[0];
          12'h008: ier       <= w_q[1:0];
          12'h00C: isr       <= isr ^ w_q[1:0];
          12'h010: base_lo   <= merge(base_lo,   w_q, wstrb_q);
          12'h014: base_hi   <= merge(base_hi,   w_q, wstrb_q);
          12'h018: n_instr_q <= merge(n_instr_q, w_q, wstrb_q);
          12'h01C: pos_q     <= merge(pos_q,     w_q, wstrb_q);
          12'h024: tok_q     <= merge(tok_q,     w_q, wstrb_q);
          default: if (aw_q[11:8] == 4'h1) begin
            if (aw_q[2]) base_q[aw_q[7:3]][63:32] <= merge(base_q[aw_q[7:3]][63:32], w_q, wstrb_q);
            else         base_q[aw_q[7:3]][31:0]  <= merge(base_q[aw_q[7:3]][31:0],  w_q, wstrb_q);
          end
        endcase
      end

      if (busy) cycles <= cycles + 1;

      // ---- read channel
      if (rvalid && rready) rvalid <= 1'b0;
      if (do_rd) begin
        rvalid <= 1'b1;
        case (araddr)
          12'h000: begin
                     rdata <= {28'd0, ready_bit, !busy, done_bit, start_bit};
                     done_bit <= 1'b0; ready_bit <= 1'b0;          // clear-on-read
                   end
          12'h004: rdata <= {31'd0, gie};
          12'h008: rdata <= {30'd0, ier};
          12'h00C: rdata <= {30'd0, isr};
          12'h010: rdata <= base_lo;
          12'h014: rdata <= base_hi;
          12'h018: rdata <= n_instr_q;
          12'h01C: rdata <= pos_q;
          12'h020: rdata <= result_token;
          12'h024: rdata <= tok_q;
          12'h040: rdata <= cycles[31:0];
          12'h044: rdata <= cycles[63:32];
          12'h048: rdata <= perf.ws_beats[31:0];
          12'h04C: rdata <= 32'(perf.ws_beats[47:32]);
          12'h050: rdata <= perf.ws_active[31:0];
          12'h054: rdata <= 32'(perf.ws_active[47:32]);
          12'h058: rdata <= perf.ws_starve[31:0];
          12'h05C: rdata <= 32'(perf.ws_starve[47:32]);
          12'h060: rdata <= perf.ws_stall[31:0];
          12'h064: rdata <= 32'(perf.ws_stall[47:32]);
          12'h068: rdata <= {31'd0, perf.axi_err};
          default: rdata <= (araddr[11:8] == 4'h1)
                             ? (araddr[2] ? base_q[araddr[7:3]][63:32] : base_q[araddr[7:3]][31:0])
                             : 32'hDEAD_BEEF;
        endcase
      end

      // ---- kernel done (after the read block: a same-cycle read of 0x00 must not lose it)
      if (ap_done && busy) begin
        busy <= 1'b0; start_bit <= 1'b0; done_bit <= 1'b1; ready_bit <= 1'b1;
        if (ier[0]) isr[0] <= 1'b1;
        if (ier[1]) isr[1] <= 1'b1;
      end
    end
  end
endmodule