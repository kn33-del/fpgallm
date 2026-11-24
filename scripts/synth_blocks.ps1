# Synthesize each LLM block out-of-context, one after another (fits in 32 GB RAM).
#   powershell -File scripts/synth_blocks.ps1 [LLM_TINYLLAMA]
param([string]$Define = "LLM_TINYLLAMA",
      [string[]]$Blocks = @("llm_ctrl", "mlp_unit", "attn_unit", "mm_engine", "mem_subsys"))
$vivado = "C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
foreach ($b in $Blocks) {
  New-Item -ItemType Directory -Force "build\synth\$b" | Out-Null
  $t0 = Get-Date
  Write-Output "=== $b start $($t0.ToString('HH:mm'))"
  & $vivado -mode batch -nojournal -log "build\synth\$b\vivado.log" -source scripts\synth_ooc.tcl -tclargs synth $b $Define | Out-Null
  $ok = Select-String -Quiet -Path "build\synth\$b\vivado.log" -Pattern "^DONE synth"
  $min = [int]((Get-Date) - $t0).TotalMinutes
  Write-Output "=== $b $(if ($ok) {'OK'} else {'FAILED'}) in $min min"
}
Write-Output "ALL BLOCKS FINISHED"
