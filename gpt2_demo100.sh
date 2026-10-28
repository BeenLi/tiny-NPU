#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh" >/dev/null 2>&1
cd "$ROOT/sim/verilator" || exit 1
LOG="$ROOT/gpt2_demo100.log"
rm -f "$ROOT/gpt2_demo100.DONE" "$ROOT/gpt2_demo100.FAIL"
exec >"$LOG" 2>&1
echo "[$(date)] start; python=$(command -v python) HF_ENDPOINT=${HF_ENDPOINT:-<unset>}"
if bash run_demo.sh --prompt "good mornining" --max-tokens 100; then
  touch "$ROOT/gpt2_demo100.DONE"
else
  touch "$ROOT/gpt2_demo100.FAIL"
fi
echo "[$(date)] exit rc=$?"
