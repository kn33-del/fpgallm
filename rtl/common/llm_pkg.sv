// =============================================================================
// llm_pkg.sv -- shared params, types, and CROSS-TEAM CONTRACTS.
// OWNER: integration. Every team imports this. Changes = PR reviewed by all leads.
// =============================================================================
package llm_pkg;

  // ---------------------------------------------------------------------------
  // Model dims. Defaults = Llama-3.2-1B. Override for other models.
  // ---------------------------------------------------------------------------
`ifdef LLM_TINY
  // Simulation-only shrunken model (define LLM_TINY). Same structure/contracts, tiny sizes,
  // so the full-pipeline testbench finishes in minutes. Never used for synthesis.
  parameter int D_MODEL    = 128;
  parameter int N_HEADS    = 2;
  parameter int N_KV_HEADS = 1;
  parameter int HEAD_DIM   = 64;
  parameter int D_FF       = 256;
  parameter int N_LAYERS   = 2;
  parameter int VOCAB      = 256;
  parameter int MAX_SEQ    = 64;
`elsif LLM_TINYLLAMA
  // TinyLlama-1.1B-Chat-v1.0 (open stand-in while Llama-3.2-1B access is pending).
  // Same Llama architecture, head_dim and hidden size; GQA 32/4; fits all contracts as-is.
  parameter int D_MODEL    = 2048;
  parameter int N_HEADS    = 32;
  parameter int N_KV_HEADS = 4;
  parameter int HEAD_DIM   = 64;
  parameter int D_FF       = 5632;
  parameter int N_LAYERS   = 22;
  parameter int VOCAB      = 32000;
  parameter int MAX_SEQ    = 2048;
`else
  parameter int D_MODEL    = 2048;
  parameter int N_HEADS    = 32;
  parameter int N_KV_HEADS = 8;        // GQA: N_HEADS/N_KV_HEADS = 4 Q heads per KV head
  parameter int HEAD_DIM   = 64;
  parameter int D_FF       = 8192;
  parameter int N_LAYERS   = 16;
  parameter int VOCAB      = 128256;
  parameter int MAX_SEQ    = 2048;
`endif

  // ---------------------------------------------------------------------------
  // Numerics. Baseline = W8A8, symmetric, per-output-channel weight scales,
  // per-tensor activation scale. Vector units run in VEC_W fixed point.
  // The golden model (python) is the spec for Q-points, rounding, saturation.
  // RTL teams do NOT invent formats -- they match the golden model bit-exactly.
  // ---------------------------------------------------------------------------
  parameter int DATA_W  = 8;
  parameter int ACC_W   = 32;
  parameter int VEC_W   = 16;
  parameter int SCALE_W = 32;          // requant = (acc * mult) >>> shift, TFLite-style

  // T3 vec-unit output quantizer (RMSNorm and SiLU*mul int8 outputs):
  //   y = sat8((value * T3_QUANT_MULT) >>> T3_QUANT_SHIFT). Set by the checkpoint quantizer
  //   (verif/llmtb/quantize.py); RMSNorm gammas absorb it, so only SiLU*mul sees it.
  // T1 requant: out = round(acc * mult / 2^T1_REQUANT_SHIFT). Real checkpoints need more than
  // 16 bits so small per-channel multipliers keep precision (quantizer derives mult from this).
`ifdef LLM_TINYLLAMA
  parameter int T1_REQUANT_SHIFT = 28;
  parameter int T3_QUANT_MULT  = 1;    // TODO(quantizer): fill from calibration
  parameter int T3_QUANT_SHIFT = 8;
`else
  parameter int T1_REQUANT_SHIFT = 16;
  parameter int T3_QUANT_MULT  = 1;
  parameter int T3_QUANT_SHIFT = 0;
`endif

  // ---------------------------------------------------------------------------
  // Compute array. ARR_R = K-slice depth, ARR_C = N-slice width (output channels).
  // Decode (M=1) at 1024 MAC/cycle needs 1024 fresh weight bytes/cycle
  // = 8192 bits = all 32 HBM ports x 256 b. That equality is the whole design.
  // ---------------------------------------------------------------------------
  parameter int ARR_R = 32;
  parameter int ARR_C = 32;

  // ---------------------------------------------------------------------------
  // PLANNING BUDGET (no board yet -- replace with measured numbers on bring-up)
  //   Single clock domain: kernel + HBM AXI ports both at KERNEL_MHZ. No 450 MHz domain in baseline.
  //   Port ceiling   = 32 ports x 32 B x 300 MHz = 307 GB/s
  //   Planning BW    = 280 GB/s (~91% of port ceiling; ~61% of 460 GB/s HBM spec)
  //   Array demand   = ARR_R*ARR_C B/cycle x 300 MHz = 307 GB/s  -> array matched to ports, keep 32x32
  //   Expected decode, Llama-3.2-1B W8 (~1.24 GB/token): 280 / 1.24 ~= 225 tok/s
  //   Utilization metric we report: delivered weight GB/s / PORT_CEIL_GBPS (say so explicitly).
  //   Stretch (post-baseline): streamer at 450 MHz + async FIFO -> ~400 GB/s, ~1.4x.
  // ---------------------------------------------------------------------------
  parameter int KERNEL_MHZ     = 300;
  parameter int PORT_CEIL_GBPS = 307;
  parameter int PLAN_HBM_GBPS  = 280;

  // ---------------------------------------------------------------------------
  // Memory
  // ---------------------------------------------------------------------------
  parameter int AXI_DW    = 256;                 // per HBM AXI port
  parameter int AXI_AW    = 34;                  // 16 GB  (kernel-internal {PC[4:0], off[28:0]})
  parameter int M_AXI_AW  = 64;                  // kernel m_axi address width (Vitis)
  parameter int AXI_IDW   = 6;
  parameter int N_HBM     = 32;                  // pseudo-channels / AXI ports used
  parameter int BEAT_W    = AXI_DW;              // narrow client port
  parameter int WSTREAM_W = N_HBM * AXI_DW;      // wide weight/KV port = 8192 b

  // ---------------------------------------------------------------------------
  // CONTRACT: unified address space. Every client request carries addr_t;
  // T4's router steers by region. Clients never care where data physically is.
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {
    RGN_HBM     = 2'd0,   // off-chip HBM (weights, scales, gammas, KV cache, embeddings)
    RGN_ACT     = 2'd1,   // on-chip activation buffer (URAM)
    RGN_SCRATCH = 2'd2,   // on-chip scratch (attn scores/probs, MLP intermediates)
    RGN_RSVD    = 2'd3
  } region_e;

  typedef struct packed {
    region_e           rgn;
    logic [AXI_AW-1:0] off;  // byte offset within region
  } addr_t;

  // 2D strided request: covers row-major tensors and tiles. len in beats of the port width.
  typedef struct packed {
    addr_t       base;
    logic [15:0] row_beats;
    logic [15:0] n_rows;
    logic [31:0] row_stride;   // bytes between row starts
  } mem_req_t;

  // ---------------------------------------------------------------------------
  // CONTRACT: WEIGHT LAYOUT (produced offline by the packer script, owner T4).
  //   W is K x N (y = x W). Tiled into 32x32 int8 tiles. Order: n_tile outer, k_tile inner.
  //   One wide beat (8192 b) = one full tile.
  //   Within a tile, row r (32 int8 along N) lives in HBM PC r. => PC r holds all rows k = r mod 32.
  //   Each port reads ONLY its own PC (no HBM crossbar traffic on the weight stream).
  //   Per-output-channel scales: separate array, ARR_C scales per n_tile, contiguous.
  //   Fused QKV: Wqkv = [Wq | Wk | Wv] along N (N = 2048+512+512).
  //   Fused gate/up: [Wg | Wu] along N, interleaved per n_tile so silu_mul can stream pairs.
  //
  // CONTRACT: KV CACHE LAYOUT (T2 + T4 must agree; T4 implements).
  //   Base per (layer, kv_head). Both stored as B-operands in the SAME tile format above:
  //   K stored TRANSPOSED: K^T is [HEAD_DIM x MAX_SEQ]  (scores = q . K^T -> K=HEAD_DIM, N=pos+1)
  //   V stored normal:     V   is [MAX_SEQ x HEAD_DIM]  (out = p . V     -> K=pos+1,  N=HEAD_DIM)
  //   int8, per-head static scale for baseline.
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // Matrix engine command (T1 implements, T2/T3/ctrl issue through mm_arb)
  //   C[M x N] = requant(A[M x K] * B[K x N])
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {
    OUT_I8    = 2'd0,   // requant to int8 (feeds another GEMM)
    OUT_VEC   = 2'd1,   // requant to VEC_W fixed point (feeds vector units / residual)
    OUT_ACC32 = 2'd2    // raw accumulators (debug / bring-up)
  } mm_out_e;

  typedef struct packed {
    addr_t       a_base;     // activations, row-major int8 (ACT/SCRATCH)
    addr_t       b_base;     // weights or KV, tile layout (HBM)
    addr_t       s_base;     // per-output-channel scales (HBM)
    addr_t       c_base;     // output (ACT/SCRATCH)
    logic [15:0] m, k, n;
    logic        b_is_kv;    // B from KV cache: n/k not multiples of 32 -> T1 must handle ragged edge
    logic [31:0] b_row_stride; // 0 = packed B; otherwise byte stride between N tiles (e.g. V cache)
    mm_out_e     out_fmt;
    logic [3:0]  tag;        // [3:2] = master id (set by mm_arb), [1:0] = master's own tag
  } mm_cmd_t;

  // ---------------------------------------------------------------------------
  // Macro-op commands from llm_ctrl
  // ---------------------------------------------------------------------------
  typedef struct packed {
    logic [7:0]  layer;
    logic [15:0] pos;        // current token position (decode); attends to [0..pos]
    addr_t       x_base;     // normed int8 input h
    addr_t       w_base;     // this layer's Wqkv; Wo at w_base + fixed offset (layout doc)
    addr_t       s_base;     // this layer's scales
    addr_t       out_base;   // VEC_W output (after Wo), consumed by RESADD
  } attn_cmd_t;

  typedef struct packed {
    logic [7:0]  layer;
    addr_t       x_base;
    addr_t       w_base;     // [Wg|Wu] then Wd
    addr_t       s_base;
    addr_t       out_base;
  } mlp_cmd_t;

  typedef enum logic [2:0] {
    V_RMSNORM  = 3'd0,  // dst(int8) = quant(src0 * rsqrt(mean(src0^2)+eps) * gamma(src1))
    V_RESADD   = 3'd1,  // dst = sat(src0 + src1)            (VEC_W)
    V_SILU_MUL = 3'd2,  // dst(int8) = quant(silu(src0) .* src1)
    V_ARGMAX   = 3'd3   // idx = argmax(src0[0..len-1])
  } vop_e;

  typedef struct packed {
    vop_e        op;
    addr_t       src0, src1, dst;
    logic [23:0] len;        // elements
  } vec_cmd_t;

  typedef struct packed {
    addr_t       src, dst;
    logic [23:0] len_beats;
  } dma_cmd_t;               // OP_EMBED (HBM row -> ACT), prefetch, spills

  // ---------------------------------------------------------------------------
  // Instruction (host-built program, one per 256-b beat)
  // ---------------------------------------------------------------------------
  typedef enum logic [7:0] {
    OP_NOP        = 8'd0,
    OP_END        = 8'd1,
    OP_EMBED      = 8'd2,   // dma token row of embedding table -> x
    OP_RMSNORM    = 8'd3,
    OP_ATTN_BLOCK = 8'd4,
    OP_RESADD     = 8'd5,
    OP_MLP_BLOCK  = 8'd6,
    OP_GEMM       = 8'd7,   // generic (LM head)
    OP_ARGMAX     = 8'd8
  } opcode_e;

  typedef struct packed {
    opcode_e     op;
    logic [7:0]  layer;
    logic [15:0] m, k, n;
    addr_t       src0, src1, src2, dst;    // meaning per opcode (see llm_ctrl decode table)
    logic [AXI_DW-8-8-48-4*$bits(addr_t)-1:0] rsvd;
  } instr_t;

  // ---------------------------------------------------------------------------
  // CONTRACT: KV CACHE ADDRESSING (T4 implements, T2 uses for GEMM b_base).
  //   Same per-PC offset in all 32 PCs. Per (layer, kvh): K^T block 4 KB/PC, then V block 4 KB/PC.
  //   K^T = [HEAD_DIM x MAX_SEQ] tiles: k_tiles=2,  n_tiles=MAX_SEQ/32 -> 128 beats/PC
  //   V   = [MAX_SEQ x HEAD_DIM] tiles: k_tiles=MAX_SEQ/32, n_tiles=2 -> 128 beats/PC
  //   T2 GEMM: b_base.off = kv_pc_off(layer, kvh, is_v)   (streamer ignores off[33:29])
  //   Streamer requests are 2D (row = one n_tile's k_tiles), so partial reads work:
  //     K^T at pos p: row_beats = 2*ceil((p+1)/32), n_rows = 1
  //     V   at pos p: row_beats = ceil((p+1)/32),   n_rows = 2, row_stride = (MAX_SEQ/32)*32 B
  //   Weights: row_beats = k_tiles, n_rows = n_tiles, row_stride = k_tiles*32 B.
  // ---------------------------------------------------------------------------
  parameter logic [28:0] KV_PC_BASE = 29'h1000_0000;     // 256 MB into each PC (weights below)
  parameter int          KV_BLK_BYTES = 4096;

  function automatic logic [28:0] kv_pc_off(input logic [7:0] layer, input logic [3:0] kvh,
                                            input logic is_v);
    logic [28:0] idx;
    idx = 29'(((32'(layer) * N_KV_HEADS + 32'(kvh)) * 2) + 32'(is_v));
    return KV_PC_BASE + (idx << 12);
  endfunction

  // kv_wr request tag: T2 puts this in kv_wr.req.base.off[28:0], rgn = RGN_HBM,
  // row_beats = HEAD_DIM/32 (= 2), n_rows = 1. Data = one head's k (or v) vector, int8.
  typedef struct packed {
    logic [7:0]  layer;
    logic [3:0]  kvh;
    logic        is_v;
    logic [15:0] pos;
  } kv_tag_t;

  // ---------------------------------------------------------------------------
  // Perf / status (T4 -> ctrl_regs). Cleared on every ap_start.
  // ---------------------------------------------------------------------------
  typedef struct packed {
    logic [47:0] ws_beats;     // 8192-b tiles delivered to T1
    logic [47:0] ws_active;    // cycles a weight stream was in progress
    logic [47:0] ws_starve;    // active, no tile available (HBM-limited)
    logic [47:0] ws_stall;     // tile available, T1 not ready (compute-limited)
    logic        axi_err;      // sticky: any non-OKAY rresp/bresp
  } perf_t;

endpackage
