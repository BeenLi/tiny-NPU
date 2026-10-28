#!/usr/bin/env bash
# Prepare the --datadir inputs for the data-driven sims, using the directory names
# that sim/verilator/regression_all.sh and run_tests.sh expect:
#  - compile every committed ONNX test model (models/*.onnx) into sim/verilator/build/graph*
#  - generate + compile the 50 fuzz models (models/fuzz/, git-ignored)
#  - random-weight LLaMA data (llama_data); GPT-2 demo_data comes from gpt2_demo.sh
# Compile failures are reported and skipped (see "已知问题" in docs/SETUP_NOTES.md).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh" >/dev/null 2>&1
cd "$ROOT" || exit 1
B=sim/verilator/build
mkdir -p "$B"
compile() {  # <datadir> <model.onnx>
  if python python/onnx_compiler/compile.py --model "$2" --outdir "$B/$1" >"$B/compile_${1//\//_}.log" 2>&1; then
    echo "OK    $1"
  else
    echo "FAIL  $1 (see $B/compile_${1//\//_}.log)"; rm -rf "${B:?}/$1"
  fi
}
while read -r d m; do compile "$d" "models/$m.onnx"; done <<'EOF'
graph mlp_32_16_8
graph_cnn conv1x_mini
graph_reduce reduce_test
graph_math math_test
graph_gather gather_test
graph_slice_concat slice_concat_test
graph_batchnorm_pool batchnorm_pool_test
graph_stress stress_test
graph_overlap_perf overlap_perf_test
graph_fp16_smoke fp16_smoke_test
graph_mixed_cnn mixed_precision_cnn_test
graph_resize_pad resize_pad_test
EOF
python python/onnx_compiler/gen_fuzz_onnx.py >"$B/gen_fuzz.log" 2>&1
for i in $(seq 0 49); do compile "graph_fuzz/case_${i}" "models/fuzz/case_${i}.onnx"; done
python python/tools/llama_gen_weights.py --outdir "$B/llama_data" >"$B/llama_gen.log" 2>&1 \
  && echo "OK    llama_data" || echo "FAIL  llama_data (see $B/llama_gen.log)"
