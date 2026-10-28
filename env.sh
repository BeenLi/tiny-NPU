#!/usr/bin/env bash
# Source this before building / running tiny-NPU:
#   source env.sh && cd sim/verilator && mkdir -p build && cd build && cmake .. && cmake --build . -j"$(nproc)"
#
# What it sets up:
#  - PATH: .conda-env/bin first -> Verilator 5.050 (conda-forge), Python 3.11, uv.
#    CMake 3.28.3 and g++ 13 come from the system (/usr/bin); CMake finds
#    .conda-env/share/verilator/verilator-config.cmake through PATH, no VERILATOR_ROOT needed.
#  - activates the Python 3.11 venv (.venv: torch CPU + transformers + onnx + numpy/scipy)
#  - huggingface.co is reachable through the host proxy; if it is ever blocked,
#    `export HF_ENDPOINT=https://hf-mirror.com` before running the demos.
_TINYNPU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PATH="$_TINYNPU_ROOT/.conda-env/bin:$PATH"
# shellcheck disable=SC1091
[ -f "$_TINYNPU_ROOT/.venv/bin/activate" ] && source "$_TINYNPU_ROOT/.venv/bin/activate"
unset _TINYNPU_ROOT
