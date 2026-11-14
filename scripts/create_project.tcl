# Recreates the Vivado project from sources in git. Project lives in build/ (gitignored).
# Usage: vivado -mode batch -source scripts/create_project.tcl
set root [file normalize [file join [file dirname [info script]] ..]]
set proj_dir [file join $root build vivado]

# U55C part (matches xilinx_u55c_gen3x16_xdma platform)
create_project llm_u55c $proj_dir -part xcu55c-fsvh2892-2L-e -force

# RTL: reference files in place (NOT copied into project) so edits land in git.
# Package + interfaces first; Vivado auto-orders the rest.
set rtl [concat \
  [glob $root/rtl/common/llm_pkg.sv] \
  [glob $root/rtl/common/llm_if.sv] \
  [glob -nocomplain $root/rtl/common/*.sv] \
  [glob -nocomplain $root/rtl/t1_compute/*.sv] \
  [glob -nocomplain $root/rtl/t2_attention/*.sv] \
  [glob -nocomplain $root/rtl/t3_mlp/*.sv] \
  [glob -nocomplain $root/rtl/t4_memory/*.sv] \
  [glob $root/rtl/llm_top.sv]]
add_files -norecurse [lsort -unique $rtl]
set_property file_type SystemVerilog [get_files -of [get_filesets sources_1] *.sv]
set_property top llm_top [get_filesets sources_1]

# Constraints
set xdc [glob -nocomplain $root/constraints/*.xdc]
if {[llength $xdc]} { add_files -fileset constrs_1 -norecurse $xdc }

# Testbenches: tb/*.sv. Set sim top per test, e.g. set_property top tb_pe [get_filesets sim_1]
set tb [glob -nocomplain $root/tb/*.sv]
if {[llength $tb]} {
  add_files -fileset sim_1 -norecurse $tb
  set_property file_type SystemVerilog [get_files -of [get_filesets sim_1] *.sv]
}

update_compile_order -fileset sources_1
puts "Project created at $proj_dir/llm_u55c.xpr"
