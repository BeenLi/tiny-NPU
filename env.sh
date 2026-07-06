#!/usr/bin/env bash
# Source this before building / running tiny-NPU:
#   source env.sh && cd sim/verilator && mkdir -p build && cd build && cmake .. && cmake --build . -j"$(nproc)"
#
# What it sets up:
#  - PATH: locally-built Verilator 5.050 + CMake 3.28.3 (~/.local), ahead of apt's
#    (verilator 4.010 / cmake 3.13.4, both too old for this repo)
#  - HF_ENDPOINT: huggingface.co is blocked on this host; use the hf-mirror.com
#    mirror so the GPT-2 / LLaMA weight downloads work
#  - activates the Python 3.11 venv (torch CPU + transformers + onnx + numpy/scipy)
export PATH="$HOME/.local/cmake-3.28.3-linux-x86_64/bin:$HOME/.local/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/.venv/bin/activate"
