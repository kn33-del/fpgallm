# make project  -> regenerate Vivado project in build/ (run after every pull that adds files)
# make gui      -> open it
# make lint     -> verilator lint (same as CI)
# make clean
RTL := rtl/common/llm_pkg.sv rtl/common/llm_if.sv \
       $(filter-out rtl/common/llm_pkg.sv rtl/common/llm_if.sv,$(wildcard rtl/common/*.sv)) \
       $(wildcard rtl/t1_compute/*.sv rtl/t2_attention/*.sv rtl/t3_mlp/*.sv rtl/t4_memory/*.sv) \
       rtl/llm_top.sv

project:
	vivado -mode batch -nojournal -nolog -source scripts/create_project.tcl

gui:
	vivado build/vivado/llm_u55c.xpr &

lint:
	verilator --lint-only -Wall -Wno-fatal -Wno-UNDRIVEN -Wno-UNUSED -Wno-PINMISSING \
	  -Wno-DECLFILENAME -Wno-MULTITOP -Wno-SYMRSVDWORD --top-module llm_top $(RTL)

clean:
	rm -rf build .Xil obj_dir *.jou *.log

.PHONY: project gui lint clean
