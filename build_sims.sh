#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh" >/dev/null 2>&1
cd "$ROOT/sim/verilator" || exit 1
LOG="$ROOT/sim_build.log"
rm -f "$ROOT/sim_build.DONE" "$ROOT/sim_build.FAIL"
exec >"$LOG" 2>&1
echo "[$(date)] cmake build start; verilator=$(command -v verilator) ($(verilator --version))"
mkdir -p build && cd build
cmake .. || { touch "$ROOT/sim_build.FAIL"; exit 1; }
cmake --build . -j"$(nproc)" || { touch "$ROOT/sim_build.FAIL"; exit 1; }
echo "[$(date)] build done. Executables:"
find . -maxdepth 1 -type f -executable | sort
touch "$ROOT/sim_build.DONE"
echo "[$(date)] exit"
