#!/usr/bin/env bash
source <repo>/env.sh >/dev/null 2>&1
cd <repo>/sim/verilator || exit 1
LOG=<repo>/sim_build.log
rm -f <repo>/sim_build.DONE <repo>/sim_build.FAIL
exec >"$LOG" 2>&1
echo "[$(date)] cmake build start; verilator=$(command -v verilator) ($(verilator --version))"
mkdir -p build && cd build
cmake .. || { touch <repo>/sim_build.FAIL; exit 1; }
cmake --build . -j"$(nproc)" || { touch <repo>/sim_build.FAIL; exit 1; }
echo "[$(date)] build done. Executables:"
find . -maxdepth 1 -type f -executable | sort
touch <repo>/sim_build.DONE
echo "[$(date)] exit"
