#!/usr/bin/env python3
"""Recalibrate encoder activation Q/DQ scales with percentile clipping.

ORT's built-in Percentile/Entropy calibrators retain histograms for every large
attention tensor and exceed the memory budget on this XLarge graph. This
small, deterministic pass keeps the same Q/DQ graph and weights as the
successful encoder MinMax model, then replaces only encoder activation scales
with symmetric p99.99 ranges measured from the same calibration images. It is
the usual percentile clipping algorithm, implemented in a streaming-friendly
way for this large graph.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quantize_int8 import preprocess, select_calibration_images  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="FP32 source ONNX")
    p.add_argument("--minmax-qdq", type=Path, required=True, help="encoder MinMax Q/DQ model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--images", type=Path, default=Path("/home/rton/val2017/val2017"))
    p.add_argument("--calibration-count", type=int, default=8)
    p.add_argument("--percentile", type=float, default=99.99)
    p.add_argument("--samples-per-tensor", type=int, default=100_000)
    args = p.parse_args()
    if not 0.0 < args.percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    source = onnx.load(str(args.source), load_external_data=True)
    qdq = onnx.load(str(args.minmax_qdq), load_external_data=True)
    initializers = {x.name: x for x in qdq.graph.initializer}
    init_names = set(initializers)
    q_nodes = [
        n for n in qdq.graph.node
        if n.op_type == "QuantizeLinear" and n.input
        and "/backbone/backbone.0/encoder/" in n.input[0]
        and n.input[0] not in init_names
    ]
    tensor_names = [n.input[0] for n in q_nodes]
    if not tensor_names:
        raise RuntimeError("no encoder activation Q nodes found")
    from polygraphy.backend.onnx import ModifyOutputs
    marked = ModifyOutputs(source, outputs=tensor_names, copy=True)()
    sess = ort.InferenceSession(marked.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [x.name for x in sess.get_outputs()]
    samples: dict[str, list[np.ndarray]] = {name: [] for name in tensor_names}
    rng = np.random.default_rng(20261104)
    paths = select_calibration_images(args.images, args.calibration_count)
    for path in paths:
        data = preprocess(path, 700, 700)
        values = sess.run(output_names, {input_name: data})
        for name, value in zip(output_names, values):
            flat = np.asarray(value, dtype=np.float32).reshape(-1)
            finite = flat[np.isfinite(flat)]
            if finite.size > args.samples_per_tensor:
                finite = finite[rng.choice(finite.size, args.samples_per_tensor, replace=False)]
            samples[name].append(finite)
    scales: dict[str, float] = {}
    for name, chunks in samples.items():
        flat = np.concatenate(chunks)
        clip = float(np.percentile(np.abs(flat), args.percentile))
        scales[name] = max(clip / 127.0, 1e-8)
    for node in q_nodes:
        scale_name = node.input[1]
        if scale_name not in initializers:
            continue
        old = numpy_helper.to_array(initializers[scale_name])
        new = np.asarray(scales[node.input[0]], dtype=old.dtype).reshape(old.shape)
        initializers[scale_name].CopyFrom(numpy_helper.from_array(new, scale_name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(qdq)
    onnx.save(qdq, str(args.output))
    print(f"wrote {args.output}; tensors={len(scales)} p{args.percentile:g}")
    print(f"calibration images={len(paths)} samples/tensor={args.samples_per_tensor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
