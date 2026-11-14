// =============================================================================
// llm_ctrl.sv -- macro-op sequencer. OWNER: integration (assign one person).
// Host writes the program ONCE into HBM; per token it only writes token_pos + ap_start.
// =============================================================================
module llm_ctrl import llm_pkg::*; (
  input  logic        clk, rst_n,

  // ctrl_regs (T4)
  input  logic        start,          // ap_start pulse
  output logic        done,           // ap_done pulse
  input  addr_t       instr_base,
  input  logic [15:0] n_instr,
  input  logic [15:0] token_pos,      // overrides any pos in the program
  output logic [31:0] result_token,   // argmax result, readable by host

  mem_rd_if.client    ifetch,         // 1 beat = 1 instr_t
  mm_if.master        mm,             // OP_GEMM (LM head)

  // T2
  output attn_cmd_t   attn_cmd,
  output logic        attn_cmd_valid,
  input  logic        attn_cmd_ready,
  input  logic        attn_done,

  // T3
  output mlp_cmd_t    mlp_cmd,
  output logic        mlp_cmd_valid,
  input  logic        mlp_cmd_ready,
  input  logic        mlp_done,
  output vec_cmd_t    vec_cmd,
  output logic        vec_cmd_valid,
  input  logic        vec_cmd_ready,
  input  logic        vec_done,
  input  logic [31:0] argmax_idx,

  // T4
  output dma_cmd_t    dma_cmd,
  output logic        dma_cmd_valid,
  input  logic        dma_cmd_ready,
  input  logic        dma_done
);

  // FSM: IDLE -> FETCH -> DECODE -> ISSUE -> WAIT(done of that unit) -> FETCH ... OP_END -> DONE
  //
  // Fetch: one mem_req_t for the whole program (row_beats = n_instr), FIFO the beats,
  //        pop one instr per DECODE. Program is tiny (~100 instrs), so fetch it all up front.
  //
  // Decode table (src/dst meaning per op):
  //   OP_EMBED     : dma src = src0 + token_id*D_MODEL*2 (token_id from src1 field or prev argmax),
  //                  dst = dst. NOTE: for autoregressive decode, token_id = last result_token.
  //   OP_RMSNORM   : vec V_RMSNORM, src0 = x, src1 = gamma (HBM), dst = h, len = D_MODEL
  //   OP_ATTN_BLOCK: attn_cmd{layer, pos=token_pos, x=src0, w=src1, s=src2, out=dst}
  //   OP_RESADD    : vec V_RESADD, src0 = x, src1 = a/m, dst = x
  //   OP_MLP_BLOCK : mlp_cmd{layer, x=src0, w=src1, s=src2, out=dst}
  //   OP_GEMM      : mm cmd {a=src0, b=src1, s=src2, c=dst, m,k,n, OUT_VEC}
  //   OP_ARGMAX    : vec V_ARGMAX, src0 = logits, len = n; latch argmax_idx -> result_token
  //   OP_END       : pulse done
  //
  // Hazards: baseline waits for each unit's done before next issue -> no scoreboard needed.
  // Later: overlap next layer's weight prefetch (dma) with current vector ops.

  // TODO: static check instr_t is exactly one beat
  // initial assert ($bits(instr_t) == AXI_DW);

  // TODO: perf counters (cycles per opcode, mm busy cycles, HBM bytes) -> ctrl_regs.
  //       You need these to claim a utilization number.
  //       Report: weight bytes / total cycles vs PORT_CEIL_GBPS. Gap between per-GEMM BW (~280)
  //       and whole-token BW = overhead from vector ops / attention / cmd gaps -> overlap target.

endmodule
