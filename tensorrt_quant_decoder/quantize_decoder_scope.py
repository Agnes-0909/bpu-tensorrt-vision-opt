#!/usr/bin/env python3
"""Quantize one explicit decoder scope on top of the SmoothQuant encoder graph.

This wrapper intentionally uses include patterns instead of excluding /backbone/.
The input graph already contains encoder Q/DQ nodes; only the selected decoder
MatMul/Gemm nodes are sent through ORT static QDQ quantization.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUANTIZE = ROOT / "tensort_quant" / "quantize_int8.py"
DEFAULT_SOURCE = ROOT / "tensort_quant" / "artifacts_xlarge_smoothquant_256" / "rfdetr-xlarge-smoothquant-256.onnx"
DEFAULT_IMAGES = Path("/home/rton/val2017/val2017")
OUT_ROOT = Path(__file__).resolve().parent / "artifacts"


def patterns(scope: str) -> list[str]:
    def each(suffix: str) -> list[str]:
        return [f"/transformer/decoder/layers.{layer}/{suffix}" for layer in range(5)]

    qkv = each("self_attn/MatMul") + each("self_attn/MatMul_1") + each("self_attn/MatMul_2")
    self_out = each("self_attn/Gemm")
    ffn = each("linear1/MatMul") + each("linear2/MatMul")
    ffn_fc1 = each("linear1/MatMul")
    ffn_fc2 = each("linear2/MatMul")
    cross = each("cross_attn/value_proj/MatMul") + each("cross_attn/output_proj/MatMul")
    attention_weights = each("cross_attn/attention_weights/MatMul")
    offsets = each("cross_attn/sampling_offsets/MatMul")
    scopes = {
        "self_qkv": qkv,
        "self_qkvo": qkv + self_out,
        "ffn": ffn,
        "ffn_fc1": ffn_fc1,
        "ffn_fc2": ffn_fc2,
        "cross_value_output": cross,
        "cross_attention_weights": attention_weights,
        "cross_sampling_offsets": offsets,
        "selective": qkv + self_out + ffn + cross,
        "all_linear": qkv + self_out + ffn + cross,
    }
    if scope not in scopes:
        raise ValueError(f"unknown scope {scope}; choose from {', '.join(scopes)}")
    return scopes[scope]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=[
        "self_qkv", "self_qkvo", "ffn", "cross_value_output",
        "ffn_fc1", "ffn_fc2",
        "cross_attention_weights", "cross_sampling_offsets", "selective", "all_linear",
    ])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--calibration-method", choices=["MinMax", "Percentile"], default="MinMax")
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--calibration-cache", type=Path, default=None)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if not args.images.exists():
        raise FileNotFoundError(args.images)
    if args.output is None:
        args.output = OUT_ROOT / f"{args.scope}_{args.calibration_method.lower()}" / f"rfdetr-xlarge-decoder-{args.scope.lower()}-{args.calibration_method.lower()}.onnx"
    if args.manifest is None:
        args.manifest = args.output.parent / "calibration_manifest.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    include = patterns(args.scope)
    command = [
        sys.executable, str(QUANTIZE),
        "--model", str(args.source),
        "--images", str(args.images),
        "--output", str(args.output),
        "--manifest", str(args.manifest),
        "--calibration-count", str(args.count),
        "--quantize-ops", "MatMul,Gemm",
        "--calibration-method", args.calibration_method,
        "--force-node-selection",
    ]
    for item in include:
        command.extend(["--node-name", item])
    if args.calibration_cache is not None:
        command.extend(["--calibration-cache", str(args.calibration_cache)])
    if args.calibration_method == "Percentile":
        command.extend(["--calib-percentile", str(args.percentile)])
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest.update({
        "decoder_scope": args.scope,
        "decoder_include_patterns": include,
        "encoder_source_is_existing_smoothquant_qdq": True,
        "source_encoder_model": str(args.source),
    })
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
