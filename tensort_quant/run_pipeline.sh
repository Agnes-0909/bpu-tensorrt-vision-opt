#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/rton/miniconda3/envs/torch/bin/python}"
images="${COCO_IMAGES:-/home/rton/val2017/val2017}"
annotations="${COCO_ANNOTATION:-/home/rton/annotations/instances_val2017.json}"
calibration_count="${CALIBRATION_COUNT:-64}"
limit="${COCO_LIMIT:-0}"
iterations="${BENCHMARK_ITERATIONS:-500}"
warmup="${BENCHMARK_WARMUP:-50}"
model_onnx="${MODEL_ONNX:-${repo_dir}/python_infer/output/rfdetr-xlarge.onnx}"
fp32_engine="${FP32_ENGINE:-${repo_dir}/model_xlarge.engine}"
artifacts="${ARTIFACTS_DIR:-${script_dir}/artifacts_xlarge_conv}"

mkdir -p "${artifacts}"
if [[ ! -f "${fp32_engine}" ]]; then
  trtexec_bin="${TRTEXEC:-/home/rton/TensorRT/TensorRT-11.4.0.106/bin/trtexec}"
  "${trtexec_bin}" \
    "--onnx=${model_onnx}" \
    "--saveEngine=${fp32_engine}" --skipInference
fi
"${python_bin}" "${script_dir}/quantize_int8.py" \
  --model "${model_onnx}" \
  --images "${images}" \
  --output "${artifacts}/rfdetr-xlarge-conv-int8.onnx" \
  --manifest "${artifacts}/calibration_manifest.json" \
  --calibration-count "${calibration_count}" --quantize-ops Conv
"${python_bin}" "${script_dir}/build_int8_engine.py" \
  --onnx "${artifacts}/rfdetr-xlarge-conv-int8.onnx" \
  --engine "${artifacts}/rfdetr-xlarge-conv-int8.engine"

if [[ ! -x "${repo_dir}/tensorrt_infer/build/rfdetr_benchmark" ]]; then
  (cd "${repo_dir}/tensorrt_infer" && ./build.sh)
fi

echo "=== FP32 TensorRT benchmark ==="
"${repo_dir}/tensorrt_infer/build/rfdetr_benchmark" \
  "${fp32_engine}" "${repo_dir}/img/000000085772.jpg" "${iterations}" "${warmup}" 0.5 -1 \
  | tee "${artifacts}/benchmark_fp32.txt"
echo "=== INT8 TensorRT benchmark ==="
"${repo_dir}/tensorrt_infer/build/rfdetr_benchmark" \
  "${artifacts}/rfdetr-xlarge-conv-int8.engine" "${repo_dir}/img/000000085772.jpg" "${iterations}" "${warmup}" 0.5 -1 \
  | tee "${artifacts}/benchmark_int8.txt"

echo "=== FP32 COCO evaluation ==="
"${python_bin}" "${repo_dir}/evaluate_coco_tensorrt.py" \
  --engine "${fp32_engine}" --images "${images}" --annotation "${annotations}" \
  --limit "${limit}" --output "${artifacts}/coco_predictions_fp32.json" --report "${artifacts}/coco_fp32.json" \
  | tee "${artifacts}/evaluate_fp32.txt"
echo "=== INT8 COCO evaluation ==="
"${python_bin}" "${repo_dir}/evaluate_coco_tensorrt.py" \
  --engine "${artifacts}/rfdetr-xlarge-conv-int8.engine" --images "${images}" --annotation "${annotations}" \
  --limit "${limit}" --output "${artifacts}/coco_predictions_int8.json" --report "${artifacts}/coco_int8.json" \
  | tee "${artifacts}/evaluate_int8.txt"

"${python_bin}" - "${artifacts}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
fp = json.loads((root / "coco_fp32.json").read_text())
iq = json.loads((root / "coco_int8.json").read_text())
def delta(key): return iq["metrics"][key] - fp["metrics"][key]
summary = {
  "fp32": fp["metrics"], "int8": iq["metrics"],
  "delta": {key: delta(key) for key in fp["metrics"]},
  "speedup_inference_mean": fp["timing_ms"]["inference"]["mean_ms"] / iq["timing_ms"]["inference"]["mean_ms"],
}
(root / "comparison.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
