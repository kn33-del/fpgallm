# make project  -> regenerate Vivado project in build/ (run after every pull that adds files)
# make gui      -> open it
# make lint     -> verilator lint (same as CI)
# make test-t2  -> simulate decode attention
# make verif    -> pyuvm/cocotb regression: T1..T4 unit benches + full pipeline (see verif/README.md)
#                  make verif BENCH="t1 t3"  /  make verif BENCH=top ARGS="-k TopDecode3Tokens"
# make clean
RTL := rtl/common/llm_pkg.sv rtl/common/llm_if.sv \
       $(filter-out rtl/common/llm_pkg.sv rtl/common/llm_if.sv,$(wildcard rtl/common/*.sv)) \
       $(wildcard rtl/t1_compute/*.sv rtl/t2_attention/*.sv rtl/t3_mlp/*.sv rtl/t4_memory/*.sv) \
       rtl/llm_top.sv

T2_TEST := build/tb_t2_attention/Vtb_t2_attention
T2_SOURCES := rtl/common/llm_pkg.sv rtl/common/llm_if.sv \
              rtl/t2_attention/t2_attention.sv tb/tb_t2_attention.sv

project:
	vivado -mode batch -nojournal -nolog -source scripts/create_project.tcl

gui:
	vivado build/vivado/llm_u55c.xpr &

lint:
	verilator --lint-only -Wall -Wno-fatal -Wno-UNDRIVEN -Wno-UNUSED -Wno-PINMISSING \
	  -Wno-DECLFILENAME -Wno-MULTITOP -Wno-SYMRSVDWORD --top-module llm_top $(RTL)

test-t2: $(T2_TEST)
	$<

$(T2_TEST): $(T2_SOURCES)
	mkdir -p $(@D)
	verilator --binary --timing --assert -j 4 -Wno-TIMESCALEMOD \
	  --Mdir $(@D) --top-module tb_t2_attention $^

verif:
	python3 verif/run.py $(BENCH) $(ARGS)

clean:
	rm -rf build .Xil obj_dir *.jou *.log

.PHONY: project gui lint test-t2 verif clean
