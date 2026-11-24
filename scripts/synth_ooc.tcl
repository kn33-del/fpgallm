# Out-of-context synthesis (+ optional place/route) of llm_top for the Alveo U55C part.
#   vivado -mode batch -nojournal -nolog -source scripts/synth_ooc.tcl -tclargs [synth|impl]
# Reports land in build/synth/. No platform/shell needed: this checks that the RTL synthesizes,
# how big it is, and whether it can close 300 MHz before any Vitis link is attempted.
set mode [expr {[llength $argv] > 0 ? [lindex $argv 0] : "synth"}]
set part xcu55c-fsvh2892-2L-e
set root [file normalize [file dirname [info script]]/..]
set out  $root/build/synth
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

set xdc $out/clk.xdc
set f [open $xdc w]
puts $f "create_clock -name ap_clk -period 3.333 \[get_ports ap_clk\]"
close $f
read_xdc -mode out_of_context $xdc

synth_design -top llm_top -mode out_of_context -flatten_hierarchy rebuilt
write_checkpoint -force $out/post_synth.dcp
report_utilization -file $out/util_synth.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_synth_hier.rpt
report_timing_summary -max_paths 20 -file $out/timing_synth.rpt
report_methodology -file $out/methodology_synth.rpt

if {$mode eq "impl"} {
  opt_design
  place_design
  phys_opt_design
  route_design
  write_checkpoint -force $out/post_route.dcp
  report_utilization -file $out/util_route.rpt
  report_timing_summary -max_paths 20 -file $out/timing_route.rpt
}
puts "DONE $mode"
