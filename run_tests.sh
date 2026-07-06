#!/usr/bin/env bash
source <repo>/env.sh >/dev/null 2>&1
cd <repo>/sim/verilator/build || exit 1
LOG=<repo>/test_results.log
rm -f <repo>/test_results.DONE
exec >"$LOG" 2>&1
SIMS="npu_sim engine_sim integration_sim gpt2_block_sim kv_cache_sim \
onnx_smoke_sim onnx_cnn_smoke_sim onnx_reduce_sim onnx_math_sim onnx_gather_sim \
onnx_slice_concat_sim onnx_batchnorm_pool_sim onnx_fuzz_sim onnx_stress_sim \
onnx_overlap_perf_sim onnx_fp16_smoke_sim mixed_precision_cnn_sim onnx_resize_pad_sim"
pass=0; fail=0
for s in $SIMS; do
  if [ ! -x ./$s ]; then echo "MISSING   $s"; fail=$((fail+1)); continue; fi
  out=$(timeout 180 ./$s 2>&1); rc=$?
  last=$(printf "%s" "$out" | grep -iE "pass|fail|error|mismatch|match|OK|complete|✓|✗" | tail -1 | cut -c1-90)
  if [ $rc -eq 0 ]; then echo "PASS      $s | ${last}"; pass=$((pass+1));
  else echo "FAIL(rc=$rc) $s | ${last}"; fail=$((fail+1)); fi
done
echo "==== SUMMARY: ${pass} passed, ${fail} failed ===="
touch <repo>/test_results.DONE
