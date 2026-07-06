#!/usr/bin/env bash
source <repo>/env.sh >/dev/null 2>&1
cd <repo>/sim/verilator || exit 1
LOG=<repo>/gpt2_demo.log
rm -f <repo>/gpt2_demo.DONE <repo>/gpt2_demo.FAIL
exec >"$LOG" 2>&1
echo "[$(date)] GPT-2 demo start; python=$(command -v python); HF_ENDPOINT=$HF_ENDPOINT"
if bash ./run_demo.sh --prompt "Hello" --max-tokens 10; then
  touch <repo>/gpt2_demo.DONE
else
  touch <repo>/gpt2_demo.FAIL
fi
echo "[$(date)] GPT-2 demo exit"
