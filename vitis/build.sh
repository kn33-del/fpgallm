#!/usr/bin/env bash
# One-command U55C bitstream build: RTL -> llm_kernel.xo -> llm_kernel.xclbin
#
#   source <Vitis install>/settings64.sh      # same Vitis/Vivado version as the platform/XRT
#   ./vitis/build.sh [platform] [model-define]
#
# platform     default xilinx_u55c_gen3x16_xdma_3_202210_1 (see: platforminfo -l)
# model-define default LLM_TINYLLAMA (dims + quant params baked into llm_pkg)
# Needs Linux, ~64 GB RAM, several hours. Output: build/llm_kernel.xclbin + build/vpp/ logs.
set -euo pipefail
PLAT=${1:-xilinx_u55c_gen3x16_xdma_3_202210_1}
MODEL=${2:-LLM_TINYLLAMA}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

for t in vivado v++ platforminfo; do
  command -v $t >/dev/null || { echo "ERROR: $t not on PATH - source Vitis settings64.sh first"; exit 1; }
done
platforminfo -p "$PLAT" >/dev/null 2>&1 || {
  echo "ERROR: platform $PLAT not found. Installed platforms:"; platforminfo -l | grep -i '"name"' || true; exit 1; }
echo "[build] $(v++ --version | head -1)  platform=$PLAT  model=$MODEL"

mkdir -p build
echo "[build] 1/2 packaging RTL kernel -> build/xo/llm_kernel.xo"
vivado -mode batch -nojournal -log build/package_xo.log -source vitis/package_xo.tcl -tclargs "$MODEL"

echo "[build] 2/2 v++ link (synthesis + place + route + bitstream; hours) -> build/llm_kernel.xclbin"
v++ -l -t hw --platform "$PLAT" \
    --config vitis/u55c.cfg --config vitis/vivado.cfg \
    --save-temps --temp_dir build/vpp --log_dir build/vpp/logs --report_dir build/vpp/reports \
    -o build/llm_kernel.xclbin build/xo/llm_kernel.xo
echo "[build] DONE: build/llm_kernel.xclbin"
echo "[build] timing: see build/vpp/reports/link/imp/*timing_summary*  (v++ lowers the clock if 300 MHz fails)"
