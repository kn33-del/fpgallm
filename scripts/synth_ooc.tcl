# Out-of-context synthesis (+ optional place/route) for the Alveo U55C part.
#   vivado -mode batch -nojournal -source scripts/synth_ooc.tcl -tclargs [synth|impl] [top] [define]
#     top    : llm_top (default) or one block: mm_engine attn_unit mlp_unit mem_subsys llm_ctrl
#     define : model-size macro, e.g. LLM_TINYLLAMA (default: none = llm_pkg default dims)
# Reports land in build/synth/<top>/. No platform/shell needed: this checks that the RTL
# synthesizes, how big it is, and whether it can close 300 MHz before any Vitis link.
# The full llm_top needs >32 GB RAM; per-block runs fit on a laptop.
set mode   [expr {[llength $argv] > 0 ? [lindex $argv 0] : "synth"}]
set top    [expr {[llength $argv] > 1 ? [lindex $argv 1] : "llm_top"}]
set define [expr {[llength $argv] > 2 ? [lindex $argv 2] : ""}]
set part xcu55c-fsvh2892-2L-e
set root [file normalize [file dirname [info script]]/..]
set out  $root/build/synth/$top
file mkdir $out

if {[llength [get_parts -quiet $part]] == 0} {
  puts "ERROR: part $part not installed in this Vivado"; exit 2
}
set_part $part
set_param general.maxThreads 8
# Big inferred RAMs (mm_accum rows = 2 Mb, act_buf = 4 Mb) exceed Vivado's 1 Mb default per variable.
set_param synth.elaboration.rodinMoreOptions {rt::set_parameter var_size_limit 8388608}

read_verilog -sv [list \
  $root/rtl/common/llm_pkg.sv $root/rtl/common/llm_if.sv $root/rtl/common/llm_ctrl.sv \
  $root/rtl/t1_compute/t1_compute.sv $root/rtl/t2_attention/t2_attention.sv \
  $root/rtl/t3_mlp/t3_mlp.sv $root/rtl/t4_memory/t4_memory.sv $root/rtl/llm_top.sv]

set clk [expr {$top eq "llm_top" ? "ap_clk" : "clk"}]
set xdc $out/clk.xdc
set f [open $xdc w]
puts $f "create_clock -name $clk -period 3.333 \[get_ports $clk\]"
close $f
read_xdc -mode out_of_context $xdc

set defs [expr {$define eq "" ? {} : [list -verilog_define $define]}]
# Incremental synthesis: reuse the previous run of this block and re-synthesize only the
# changed parts (big win for small edits; Vivado falls back to a full run after big rewrites).
# Needs a saved checkpoint, so it only works on a Vivado release with U55C timing data.
set ref $out/ref_synth.dcp
set incr {}
if {[file exists $out/post_synth.dcp]} {
  file copy -force $out/post_synth.dcp $ref
  read_checkpoint -incremental $ref
  set incr [list -incremental_mode default]
  puts "INFO: incremental synthesis from previous $top run"
}
synth_design -top $top -mode out_of_context -flatten_hierarchy rebuilt {*}$defs {*}$incr
report_utilization -file $out/util_synth.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_synth_hier.rpt
# Vivado 2025.2 ships xcu55c without speed files (*.smax.p2p.*; the U50 has them), so it is a
# "non-timing device": synthesis and utilization work, timing/checkpoint/place-route do not.
# Use the Vivado/Vitis release that matches the U55C platform (2022.x/2023.x) for timing.
if {[catch {report_timing_summary -max_paths 20 -file $out/timing_synth.rpt} e]} {
  puts "NOTE: no timing data for $part in this Vivado release (utilization only): $e"
  set mode synth
}
if {[catch {write_checkpoint -force $out/post_synth.dcp} e]} { puts "WARNING: no checkpoint: $e" }

if {$mode eq "impl"} {
  opt_design
  place_design
  phys_opt_design
  route_design
  write_checkpoint -force $out/post_route.dcp
  report_utilization -file $out/util_route.rpt
  report_timing_summary -max_paths 20 -file $out/timing_route.rpt
}
puts "DONE $mode $top"
