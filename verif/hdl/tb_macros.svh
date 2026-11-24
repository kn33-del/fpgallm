// Port-flattening macros for the cocotb/pyuvm wrappers.
// cocotb cannot drive SystemVerilog interface ports on the toplevel, so each wrapper
// exposes every interface as plain vectors named <prefix>_<signal>.
//   *_CLI : DUT is the client of a memory port  -> Python serves it (memory model)
//   *_SRV : DUT is the server of a memory port  -> Python is the client
//   MM_SLV: DUT receives matrix commands         -> Python is the master
//   MM_MST: DUT issues matrix commands           -> Python is the engine model
`ifndef TB_MACROS_SVH
`define TB_MACROS_SVH

`define RD_CLI_PORTS(P, W) \
  output logic [$bits(llm_pkg::mem_req_t)-1:0] P``_req, output logic P``_req_valid, \
  input  logic P``_req_ready, input logic [W-1:0] P``_data, input logic P``_valid, \
  output logic P``_ready, input logic P``_last

`define RD_CLI_CONN(P, IF) \
  assign P``_req = IF.req; assign P``_req_valid = IF.req_valid; assign IF.req_ready = P``_req_ready; \
  assign IF.data = P``_data; assign IF.valid = P``_valid; assign IF.last = P``_last; \
  assign P``_ready = IF.ready;

`define WR_CLI_PORTS(P, W) \
  output logic [$bits(llm_pkg::mem_req_t)-1:0] P``_req, output logic P``_req_valid, \
  input  logic P``_req_ready, output logic [W-1:0] P``_data, output logic P``_valid, \
  output logic P``_last, input logic P``_ready, input logic P``_done

`define WR_CLI_CONN(P, IF) \
  assign P``_req = IF.req; assign P``_req_valid = IF.req_valid; assign IF.req_ready = P``_req_ready; \
  assign P``_data = IF.data; assign P``_valid = IF.valid; assign P``_last = IF.last; \
  assign IF.ready = P``_ready; assign IF.done = P``_done;

`define RD_SRV_PORTS(P, W) \
  input  logic [$bits(llm_pkg::mem_req_t)-1:0] P``_req, input logic P``_req_valid, \
  output logic P``_req_ready, output logic [W-1:0] P``_data, output logic P``_valid, \
  input  logic P``_ready, output logic P``_last

`define RD_SRV_CONN(P, IF) \
  assign IF.req = P``_req; assign IF.req_valid = P``_req_valid; assign P``_req_ready = IF.req_ready; \
  assign P``_data = IF.data; assign P``_valid = IF.valid; assign P``_last = IF.last; \
  assign IF.ready = P``_ready;

`define WR_SRV_PORTS(P, W) \
  input  logic [$bits(llm_pkg::mem_req_t)-1:0] P``_req, input logic P``_req_valid, \
  output logic P``_req_ready, input logic [W-1:0] P``_data, input logic P``_valid, \
  input  logic P``_last, output logic P``_ready, output logic P``_done

`define WR_SRV_CONN(P, IF) \
  assign IF.req = P``_req; assign IF.req_valid = P``_req_valid; assign P``_req_ready = IF.req_ready; \
  assign IF.data = P``_data; assign IF.valid = P``_valid; assign IF.last = P``_last; \
  assign P``_ready = IF.ready; assign P``_done = IF.done;

`define MM_SLV_PORTS(P) \
  input  logic [$bits(llm_pkg::mm_cmd_t)-1:0] P``_cmd, input logic P``_cmd_valid, \
  output logic P``_cmd_ready, output logic P``_done, output logic [3:0] P``_done_tag

`define MM_SLV_CONN(P, IF) \
  assign IF.cmd = llm_pkg::mm_cmd_t'(P``_cmd); assign IF.cmd_valid = P``_cmd_valid; \
  assign P``_cmd_ready = IF.cmd_ready; assign P``_done = IF.done; assign P``_done_tag = IF.done_tag;

`define MM_MST_PORTS(P) \
  output logic [$bits(llm_pkg::mm_cmd_t)-1:0] P``_cmd, output logic P``_cmd_valid, \
  input  logic P``_cmd_ready, input logic P``_done, input logic [3:0] P``_done_tag

`define MM_MST_CONN(P, IF) \
  assign P``_cmd = IF.cmd; assign P``_cmd_valid = IF.cmd_valid; assign IF.cmd_ready = P``_cmd_ready; \
  assign IF.done = P``_done; assign IF.done_tag = P``_done_tag;

`endif
