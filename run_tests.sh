#!/usr/bin/env bash
# Batch-run the sims and summarize PASS/FAIL. Run prep_sim_data.sh first so the
# --datadir directories (same names as sim/verilator/regression_all.sh) exist.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/env.sh" >/dev/null 2>&1
cd "$ROOT/sim/verilator/build" || exit 1
LOG="$ROOT/test_results.log"
rm -f "$ROOT/test_results.DONE"
exec >"$LOG" 2>&1
# name[:datadir]
SIMS="npu_sim engine_sim integration_sim gpt2_block_sim kv_cache_sim:demo_data \
llama_block_sim:llama_data \
onnx_smoke_sim:graph onnx_cnn_smoke_sim:graph_cnn onnx_reduce_sim:graph_reduce \
onnx_math_sim:graph_math onnx_gather_sim:graph_gather onnx_slice_concat_sim:graph_slice_concat \
onnx_batchnorm_pool_sim:graph_batchnorm_pool onnx_fuzz_sim:graph_fuzz onnx_stress_sim:graph_stress \
onnx_overlap_perf_sim:graph_overlap_perf onnx_fp16_smoke_sim:graph_fp16_smoke \
mixed_precision_cnn_sim:graph_mixed_cnn onnx_resize_pad_sim:graph_resize_pad"
pass=0; fail=0
for entry in $SIMS; do
  s=${entry%%:*}; d=""; [ "$entry" != "$s" ] && d=${entry#*:}
  if [ ! -x ./$s ]; then echo "MISSING   $s"; fail=$((fail+1)); continue; fi
  if [ -n "$d" ] && [ ! -d "$d" ]; then echo "NODATA    $s (missing $d)"; fail=$((fail+1)); continue; fi
  out=$(timeout 900 ./$s ${d:+--datadir "$d"} 2>&1); rc=$?
  last=$(printf "%s" "$out" | grep -iE "pass|fail|error|mismatch|match|OK|complete|✓|✗" | tail -1 | cut -c1-90)
  if [ $rc -eq 0 ]; then echo "PASS      $s | ${last}"; pass=$((pass+1));
  else echo "FAIL(rc=$rc) $s | ${last}"; fail=$((fail+1)); fi
done
echo "==== SUMMARY: ${pass} passed, ${fail} failed ===="
touch "$ROOT/test_results.DONE"
