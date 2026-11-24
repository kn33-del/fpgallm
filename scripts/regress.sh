#!/usr/bin/env bash
# Run the pyuvm regression in WSL. From Windows PowerShell:
#   wsl.exe bash scripts/regress.sh            # everything (~15 min)
#   wsl.exe bash scripts/regress.sh t1         # one bench (t1 t2 t3 t4 top)
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1
exec python3 verif/run.py "$@"
