// =============================================================================
// llm_if.sv -- cross-team handshakes. OWNER: integration.
// All channels are AXI-Stream style: transfer when valid && ready.
// Rules: valid never depends combinationally on ready; payload stable while valid && !ready.
// =============================================================================

// Read port: client sends one mem_req_t, server returns data beats, last on final beat.
interface mem_rd_if #(parameter int W = llm_pkg::BEAT_W);
  import llm_pkg::*;
  mem_req_t     req;
  logic         req_valid, req_ready;
  logic [W-1:0] data;
  logic         valid, ready, last;
  modport client (output req, req_valid, ready, input  req_ready, data, valid, last);
  modport server (input  req, req_valid, ready, output req_ready, data, valid, last);
endinterface

// Write port: client sends req then beats. done pulses once ALL beats are committed
// (HBM B-response received / URAM written). Anyone reading that data next waits for done.
interface mem_wr_if #(parameter int W = llm_pkg::BEAT_W);
  import llm_pkg::*;
  mem_req_t     req;
  logic         req_valid, req_ready;
  logic [W-1:0] data;
  logic         valid, ready, last;
  logic         done;
  modport client (output req, req_valid, data, valid, last, input  req_ready, ready, done);
  modport server (input  req, req_valid, data, valid, last, output req_ready, ready, done);
endinterface

// Matrix engine command port. done pulses once per cmd, after C is fully written (c_wr.done).
interface mm_if;
  import llm_pkg::*;
  mm_cmd_t    cmd;
  logic       cmd_valid, cmd_ready;
  logic       done;
  logic [3:0] done_tag;
  modport master (output cmd, cmd_valid, input  cmd_ready, done, done_tag);
  modport slave  (input  cmd, cmd_valid, output cmd_ready, done, done_tag);
endinterface
