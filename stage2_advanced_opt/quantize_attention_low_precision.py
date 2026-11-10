#!/usr/bin/env python3
"""Create a diagnostic QK/AV INT8 experiment on the encoder SmoothQuant graph.

This is intentionally separate from the production quantization pipeline. It
quantizes only the encoder attention MatMul nodes selected by --scope so that
QK and AV can be evaluated independently before considering any deployment.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
QUANTIZE = ROOT / "tensort_quant" / "quantize_int8.py"
DEFAULT_MODEL = ROOT / "tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx"
DEFAULT_IMAGES = Path("/home/rton/val2017/val2017")


def attention_nodes(model: Path, scope: str) -> list[str]:
    graph = onnx.load(str(model), load_external_data=False)
    suffixes = {
        "qk": ("/attention/attention/MatMul",),
        "av": ("/attention/attention/MatMul_1",),
        "qk_av": ("/attention/attention/MatMul", "/attention/attention/MatMul_1"),
    }[scope]
    nodes = [
        node.name
        for node in graph.graph.node
        if node.op_type == "MatMul"
        and "/encoder/encoder/encoder/layer." in node.name
        and any(node.name.endswith(suffix) for suffix in suffixes)
    ]
    if len(nodes) != (12 if scope in {"qk", "av"} else 24):
        raise RuntimeError(f"expected encoder attention nodes for {scope}, found {len(nodes)}")
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["qk", "av", "qk_av"], required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.images.exists():
        raise FileNotFoundError(args.images)
    if args.count <= 0:
        raise ValueError("--count must be positive")
    nodes = attention_nodes(args.model, args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.output.parent / "calibration_manifest.json"
    command = [
        sys.executable, str(QUANTIZE),
        "--model", str(args.model), "--images", str(args.images),
        "--output", str(args.output), "--manifest", str(manifest),
        "--calibration-count", str(args.count), "--quantize-ops", "MatMul",
        "--force-node-selection",
    ]
    for node in nodes:
        command.extend(["--node-name", node])
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"scope={args.scope} nodes={len(nodes)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
