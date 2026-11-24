# Package llm_kernel as a Vitis RTL kernel (.xo). Runs in plain Vivado (Windows or Linux).
#   vivado -mode batch -nojournal -nolog -source vitis/package_xo.tcl [-tclargs <model-define>]
# <model-define> defaults to LLM_TINYLLAMA (prepended to a copy of llm_pkg.sv so the choice is
# baked into the packaged sources, independent of v++ synthesis settings).
set model [expr {[llength $argv] > 0 ? [lindex $argv 0] : "LLM_TINYLLAMA"}]
set root [file normalize [file dirname [info script]]/..]
set out  $root/build/xo
file delete -force $out
file mkdir $out/src

set fin  [open $root/rtl/common/llm_pkg.sv r]
set fout [open $out/src/llm_pkg.sv w]
if {$model ne "NONE"} { puts $fout "`define $model" }
puts -nonewline $fout [read $fin]
close $fin; close $fout

set srcs [list $out/src/llm_pkg.sv $root/rtl/common/llm_if.sv $root/rtl/common/llm_ctrl.sv \
  $root/rtl/t1_compute/t1_compute.sv $root/rtl/t2_attention/t2_attention.sv $root/rtl/t3_mlp/t3_mlp.sv \
  $root/rtl/t4_memory/t4_memory.sv $root/rtl/llm_top.sv $root/vitis/llm_kernel.sv]
foreach f [lrange $srcs 1 end] { file copy -force $f $out/src/ }

create_project -force kpack $out/kpack -part xcu55c-fsvh2892-2L-e
add_files -norecurse [lmap f $srcs {file join $out/src [file tail $f]}]
set_property top llm_kernel [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $out/ip -vendor ie497 -library kernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $out/ip/component.xml
ipx::edit_ip_in_project -upgrade true -name kedit -directory $out/kedit $out/ip/component.xml
set core [ipx::current_core]
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
set_property ipi_drc {ignore_freq_hz true} $core
set_property vitis_drc {ctrl_protocol ap_ctrl_hs} $core
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk $core
for {set p 0} {$p < 32} {incr p} {
  ipx::associate_bus_interfaces -busif [format m%02d_axi $p] -clock ap_clk $core
}
ipx::associate_bus_interfaces -clock ap_clk -reset ap_rst_n $core
set_property core_revision 1 $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -kernel $core
ipx::save_core $core
close_project -delete

package_xo -force -xo_path $out/llm_kernel.xo -kernel_name llm_kernel -ip_directory $out/ip \
  -kernel_xml $root/vitis/kernel.xml
puts "DONE: $out/llm_kernel.xo  (model $model)"
