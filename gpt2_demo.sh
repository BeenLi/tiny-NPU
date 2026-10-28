#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh" >/dev/null 2>&1
cd "$ROOT/sim/verilator" || exit 1
LOG="$ROOT/gpt2_demo.log"
rm -f "$ROOT/gpt2_demo.DONE" "$ROOT/gpt2_demo.FAIL"
exec >"$LOG" 2>&1
echo "[$(date)] GPT-2 demo start; python=$(command -v python); HF_ENDPOINT=$HF_ENDPOINT"
if bash ./run_demo.sh --prompt "Hello" --max-tokens 10; then
  touch "$ROOT/gpt2_demo.DONE"
else
  touch "$ROOT/gpt2_demo.FAIL"
fi
echo "[$(date)] GPT-2 demo exit"
