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
  input  logic [31:0] token_in,       // token id to embed this run (host writes every token)
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
  //   OP_EMBED     : dma src = src0 + token_in*D_MODEL*2, dst = dst.
  //                  Host always writes token_in (prompt token, or copies back result_token).
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
  //
  // The whole program is copied on-chip before the first issue. Streaming it instead would keep
  // T4's narrow-HBM server owned by ifetch while units need it for scales/gammas -> deadlock.

  // TODO: perf counters (cycles per opcode, mm busy cycles, HBM bytes) -> ctrl_regs.
  //       You need these to claim a utilization number.
  //       Report: weight bytes / total cycles vs PORT_CEIL_GBPS. Gap between per-GEMM BW (~280)
  //       and whole-token BW = overhead from vector ops / attention / cmd gaps -> overlap target.

  localparam int MAX_INSTR = 256;
  localparam int PW = $clog2(MAX_INSTR);

  typedef enum logic [2:0] {S_IDLE, S_FETCH_REQ, S_FETCH, S_DECODE, S_ISSUE, S_WAIT, S_DONE} st_e;
  st_e         st;
  instr_t      prog [MAX_INSTR];
  instr_t      cur;
  addr_t       base_q;
  logic [15:0] n_q, fetch_idx, pc, pos_q;
  logic [31:0] tok_q;

  // ---- instruction fetch
  assign ifetch.req       = '{base: base_q, row_beats: n_q, n_rows: 16'd1, row_stride: 32'd0};
  assign ifetch.req_valid = (st == S_FETCH_REQ);
  assign ifetch.ready     = (st == S_FETCH);

  always_ff @(posedge clk) begin
    if (st == S_FETCH && ifetch.valid) prog[PW'(fetch_idx)] <= instr_t'(ifetch.data);
  end

  // ---- command fields (combinational from the current instruction)
  always_comb begin
    dma_cmd = '{src: '{rgn: cur.src0.rgn, off: cur.src0.off + AXI_AW'(tok_q) * AXI_AW'(D_MODEL*VEC_W/8)},
                dst: cur.dst, len_beats: 24'(D_MODEL*(VEC_W/8)/(BEAT_W/8))};
    attn_cmd = '{layer: cur.layer, pos: pos_q, x_base: cur.src0, w_base: cur.src1,
                 s_base: cur.src2, out_base: cur.dst};
    mlp_cmd  = '{layer: cur.layer, x_base: cur.src0, w_base: cur.src1, s_base: cur.src2,
                 out_base: cur.dst};
    vec_cmd  = '{op: V_RMSNORM, src0: cur.src0, src1: cur.src1, dst: cur.dst, len: 24'(D_MODEL)};
    case (cur.op)
      OP_RESADD: vec_cmd.op = V_RESADD;
      OP_ARGMAX: begin vec_cmd.op = V_ARGMAX; vec_cmd.len = 24'(cur.n); end
      default: ;
    endcase
    mm.cmd = '0;
    mm.cmd.a_base = cur.src0;  mm.cmd.b_base = cur.src1;
    mm.cmd.s_base = cur.src2;  mm.cmd.c_base = cur.dst;
    mm.cmd.m = cur.m;  mm.cmd.k = cur.k;  mm.cmd.n = cur.n;
    mm.cmd.out_fmt = OUT_VEC;

    dma_cmd_valid  = (st == S_ISSUE) && (cur.op == OP_EMBED);
    attn_cmd_valid = (st == S_ISSUE) && (cur.op == OP_ATTN_BLOCK);
    mlp_cmd_valid  = (st == S_ISSUE) && (cur.op == OP_MLP_BLOCK);
    vec_cmd_valid  = (st == S_ISSUE) && (cur.op inside {OP_RMSNORM, OP_RESADD, OP_ARGMAX});
    mm.cmd_valid   = (st == S_ISSUE) && (cur.op == OP_GEMM);
  end

  logic accepted, unit_done;
  always_comb begin
    accepted  = 1'b0;
    unit_done = 1'b0;
    case (cur.op)
      OP_EMBED:      begin accepted = dma_cmd_ready;  unit_done = dma_done;  end
      OP_ATTN_BLOCK: begin accepted = attn_cmd_ready; unit_done = attn_done; end
      OP_MLP_BLOCK:  begin accepted = mlp_cmd_ready;  unit_done = mlp_done;  end
      OP_RMSNORM, OP_RESADD, OP_ARGMAX:
                     begin accepted = vec_cmd_ready;  unit_done = vec_done;  end
      OP_GEMM:       begin accepted = mm.cmd_ready;   unit_done = mm.done;   end
      default: ;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; cur <= '0; base_q <= '0; n_q <= '0; fetch_idx <= '0; pc <= '0;
      pos_q <= '0; tok_q <= '0; done <= 1'b0; result_token <= '0;
    end else begin
      done <= 1'b0;
      case (st)
        S_IDLE: if (start) begin
          base_q <= instr_base; n_q <= n_instr; pos_q <= token_pos; tok_q <= token_in;
          fetch_idx <= '0; pc <= '0;
          st <= (n_instr == 0) ? S_DONE : S_FETCH_REQ;
        end
        S_FETCH_REQ: if (ifetch.req_ready) st <= S_FETCH;
        S_FETCH: if (ifetch.valid) begin
          fetch_idx <= fetch_idx + 16'd1;
          if (ifetch.last) st <= S_DECODE;
        end
        S_DECODE: begin
          if (pc == n_q) st <= S_DONE;                 // ran off the end without OP_END
          else begin
            cur <= prog[PW'(pc)];
            pc  <= pc + 16'd1;
            st  <= S_ISSUE;
          end
        end
        S_ISSUE: begin
          case (cur.op)
            OP_END: st <= S_DONE;
            OP_EMBED, OP_RMSNORM, OP_ATTN_BLOCK, OP_RESADD, OP_MLP_BLOCK, OP_GEMM, OP_ARGMAX:
              if (accepted) st <= S_WAIT;
            default: st <= S_DECODE;                   // OP_NOP / unknown: skip
          endcase
        end
        S_WAIT: if (unit_done) begin
          if (cur.op == OP_ARGMAX) result_token <= argmax_idx;
          st <= S_DECODE;
        end
        S_DONE: begin done <= 1'b1; st <= S_IDLE; end
        default: st <= S_IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  initial assert ($bits(instr_t) == AXI_DW) else $fatal(1, "instr_t must be exactly one beat");
  always_ff @(posedge clk) if (rst_n && st == S_IDLE && start)
    assert (int'(n_instr) <= MAX_INSTR) else $fatal(1, "llm_ctrl: program longer than MAX_INSTR");
`endif

endmodule
