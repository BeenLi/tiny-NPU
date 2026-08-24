#!/usr/bin/env bash
source <repo>/env.sh >/dev/null 2>&1
# mimic the user's interactive shell (which exports a SOCKS proxy via .zshrc)
export all_proxy="socks5h://127.0.0.1:5555"; export ALL_PROXY="$all_proxy"
cd <repo>/sim/verilator || exit 1
LOG=<repo>/gpt2_demo100.log
rm -f <repo>/gpt2_demo100.DONE <repo>/gpt2_demo100.FAIL
exec >"$LOG" 2>&1
echo "[$(date)] start; all_proxy=$all_proxy HF_ENDPOINT=$HF_ENDPOINT socksio=$(python -c 'import socksio;print(socksio.__version__)' 2>&1)"
if bash run_demo.sh --prompt "good mornining" --max-tokens 100; then
  touch <repo>/gpt2_demo100.DONE
else
  touch <repo>/gpt2_demo100.FAIL
fi
echo "[$(date)] exit rc=$?"
