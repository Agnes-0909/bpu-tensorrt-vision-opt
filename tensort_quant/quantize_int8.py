#!/usr/bin/env python3
"""Create a TensorRT-compatible static INT8 Q/DQ ONNX model.

The source RF-DETR export has a fixed [1, 3, H, W] input. Calibration uses
the same RGB resize and ImageNet normalization as the TensorRT evaluator.
TensorRT 11 does not accept ONNX Runtime's INT32 bias DequantizeLinear nodes,
so those three bias nodes are folded back to equivalent FP32 initializers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def resize_torchvision_style(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match torchvision tensor resize with bilinear, antialias=False."""
    source_height, source_width = rgb.shape[:2]
    ys = (np.arange(height, dtype=np.float32) + 0.5) * source_height / height - 0.5
    xs = (np.arange(width, dtype=np.float32) + 0.5) * source_width / width - 0.5
    y_floor = np.floor(ys)
    x_floor = np.floor(xs)
    y0 = np.clip(y_floor.astype(np.int32), 0, source_height - 1)
    y1 = np.clip(y0 + 1, 0, source_height - 1)
    x0 = np.clip(x_floor.astype(np.int32), 0, source_width - 1)
    x1 = np.clip(x0 + 1, 0, source_width - 1)
    wy = np.clip(ys - y_floor, 0.0, 1.0).reshape(-1, 1, 1)
    wx = np.clip(xs - x_floor, 0.0, 1.0).reshape(1, -1, 1)
    top = rgb[y0[:, None], x0[None, :]] * (1.0 - wx) + rgb[y0[:, None], x1[None, :]] * wx
    bottom = rgb[y1[:, None], x0[None, :]] * (1.0 - wx) + rgb[y1[:, None], x1[None, :]] * wx
    return top * (1.0 - wy) + bottom * wy


def preprocess(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read calibration image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = resize_torchvision_style(rgb, width, height)
    normalized = (resized - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, paths: Iterable[Path], input_name: str, width: int, height: int):
        self.paths = list(paths)
        self.input_name = input_name
        self.width = width
        self.height = height
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.paths):
            return None
        path = self.paths[self.index]
        self.index += 1
        return {self.input_name: preprocess(path, self.width, self.height)}


def repair_int32_biases(model_path: Path) -> int:
    """Fold INT32 bias DQ nodes to FP32 constants for TensorRT's ONNX parser."""
    model = onnx.load(str(model_path))
    initializers = {item.name: item for item in model.graph.initializer}
    removed: set[str] = set()
    replacements: list[onnx.TensorProto] = []
    kept_nodes: list[onnx.NodeProto] = []
    repaired = 0

    for node in model.graph.node:
        if (
            node.op_type == "DequantizeLinear"
            and len(node.input) >= 3
            and node.input[0] in initializers
            and initializers[node.input[0]].data_type == onnx.TensorProto.INT32
        ):
            quantized = numpy_helper.to_array(initializers[node.input[0]]).astype(np.float32)
            scale = numpy_helper.to_array(initializers[node.input[1]]).astype(np.float32)
            zero_point = numpy_helper.to_array(initializers[node.input[2]]).astype(np.float32)
            value = ((quantized - zero_point) * scale).astype(np.float32)
            replacements.append(numpy_helper.from_array(value, node.output[0]))
            removed.update(node.input)
            repaired += 1
        else:
            kept_nodes.append(node)

    if repaired:
        model.graph.ClearField("node")
        model.graph.node.extend(kept_nodes)
        replacement_names = {item.name for item in replacements}
        kept_initializers = [
            item for item in model.graph.initializer
            if item.name not in removed and item.name not in replacement_names
        ]
        model.graph.ClearField("initializer")
        model.graph.initializer.extend(kept_initializers + replacements)
        onnx.checker.check_model(model)
        onnx.save(model, str(model_path))
    return repaired


def select_calibration_images(directory: Path, count: int) -> list[Path]:
    paths = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"no JPEG images found in {directory}")
    count = min(count, len(paths))
    # Uniform sampling gives a more representative calibration set than taking
    # the first files, while remaining deterministic and reproducible.
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[int(index)] for index in indices]


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/rton/rfdetr/python_infer/output/rfdetr-xlarge.onnx"))
    parser.add_argument("--images", type=Path, default=Path("/home/rton/val2017/val2017"))
    parser.add_argument("--output", type=Path, default=root / "artifacts_xlarge_conv/rfdetr-xlarge-conv-int8.onnx")
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument(
        "--calib-max-intermediate-outputs",
        type=int,
        default=0,
        help="cap intermediate outputs held by the ORT calibrator at once (useful for Entropy on large attention graphs)",
    )
    parser.add_argument(
        "--calib-percentile",
        type=float,
        default=99.99,
        help="percentile used by Percentile calibration (default: 99.99)",
    )
    parser.add_argument(
        "--calibration-method",
        choices=[method.name for method in CalibrationMethod],
        default="MinMax",
        help="activation calibration algorithm (MinMax, Entropy, Percentile, or Distribution)",
    )
    parser.add_argument("--quantize-ops", default="Conv,MatMul", help="comma-separated ONNX op types")
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help="only quantize matching node-name substrings; may be repeated",
    )
    parser.add_argument(
        "--force-node-selection",
        action="store_true",
        help="pass matching nodes through ORT's nodes_to_quantize selection (needed when the input already contains QDQ nodes)",
    )
    parser.add_argument(
        "--node-name",
        action="append",
        default=[],
        help="explicit node name to quantize; may be repeated",
    )
    parser.add_argument("--exclude-pattern", action="append", default=[], help="node-name substring to exclude")
    parser.add_argument("--manifest", type=Path, default=root / "artifacts_xlarge_conv/calibration_manifest.json")
    parser.add_argument(
        "--calibration-cache",
        type=Path,
        default=None,
        help="ORT JSON calibration cache; a matching cache avoids recollecting calibration data",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if args.calibration_count <= 0:
        raise ValueError("--calibration-count must be positive")
    quantize_ops = [item.strip() for item in args.quantize_ops.split(",") if item.strip()]
    if not quantize_ops:
        raise ValueError("--quantize-ops must contain at least one op type")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(args.model))
    if len(model.graph.input) != 1:
        raise ValueError("expected exactly one model input")
    input_name = model.graph.input[0].name
    input_dims = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    if len(input_dims) != 4 or input_dims[0] != 1 or input_dims[1] != 3 or input_dims[2] <= 0 or input_dims[3] <= 0:
        raise ValueError(f"expected static input shape [1,3,H,W], got {input_dims}")
    input_height, input_width = input_dims[2], input_dims[3]
    # Keep the final classification and box-regression heads in FP32. They
    # are numerically sensitive and contribute little to total runtime.
    nodes_to_exclude = [
        node.name
        for node in model.graph.node
        if node.op_type in {"Conv", "MatMul"}
        and any(token in node.name for token in ("class_embed", "bbox_embed"))
    ]
    nodes_to_exclude.extend(
        node.name for node in model.graph.node
        if any(pattern in node.name for pattern in args.exclude_pattern)
    )
    if args.include_pattern:
        nodes_to_exclude.extend(
            node.name
            for node in model.graph.node
            if node.op_type in quantize_ops
            and not any(pattern in node.name for pattern in args.include_pattern)
        )
    nodes_to_exclude = sorted(set(nodes_to_exclude))
    nodes_to_quantize = None
    if args.node_name:
        available = {node.name for node in model.graph.node if node.op_type in quantize_ops}
        missing = sorted(set(args.node_name) - available)
        if missing:
            raise ValueError(f"--node-name matched no quantizable nodes: {missing[:5]}")
        nodes_to_quantize = sorted(set(args.node_name))
        nodes_to_exclude.extend(
            node.name for node in model.graph.node
            if node.op_type in quantize_ops and node.name not in nodes_to_quantize
        )
        nodes_to_exclude = sorted(set(nodes_to_exclude))
    elif args.include_pattern and args.force_node_selection:
        nodes_to_quantize = sorted(
            node.name
            for node in model.graph.node
            if node.op_type in quantize_ops
            and any(pattern in node.name for pattern in args.include_pattern)
        )
        if not nodes_to_quantize:
            raise ValueError("--force-node-selection matched no quantizable nodes")
    paths = select_calibration_images(args.images, args.calibration_count)
    reader = ImageCalibrationReader(paths, input_name, input_width, input_height)
    calibration_method = CalibrationMethod[args.calibration_method]
    extra_options = {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "EnableSubgraph": True,
    }
    if args.calib_max_intermediate_outputs > 0:
        extra_options["CalibMaxIntermediateOutputs"] = args.calib_max_intermediate_outputs
    if args.calibration_method == "Percentile":
        if not 0.0 < args.calib_percentile <= 100.0:
            raise ValueError("--calib-percentile must be in (0, 100]")
        extra_options["CalibPercentile"] = args.calib_percentile
    quantize_static(
        model_input=str(args.model),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        # Quantize convolution and matrix multiplication kernels. Keeping
        # residual adds, normalization and softmax in FP32 is important for
        # the transformer decoder's numerical stability.
        op_types_to_quantize=quantize_ops,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        calibrate_method=calibration_method,
        calibration_cache_path=args.calibration_cache,
        extra_options=extra_options,
    )
    repaired = repair_int32_biases(args.output)
    args.manifest.write_text(
        json.dumps(
            {
                "source_model": str(args.model),
                "quantized_model": str(args.output),
                "input_name": input_name,
                "input_shape": input_dims,
                "calibration_images": [str(path) for path in paths],
                "calibration_count": len(paths),
                "quant_format": "QDQ",
                "activation_type": "QInt8",
                "weight_type": "QInt8",
                "per_channel": True,
                "op_types_to_quantize": quantize_ops,
                "include_patterns": args.include_pattern,
                "nodes_to_quantize": nodes_to_quantize,
                "explicit_node_names": args.node_name,
                "force_node_selection": args.force_node_selection,
                "exclude_patterns": args.exclude_pattern,
                "nodes_to_exclude": nodes_to_exclude,
                "calibration_method": args.calibration_method,
                "calib_max_intermediate_outputs": args.calib_max_intermediate_outputs,
                "calib_percentile": args.calib_percentile if args.calibration_method == "Percentile" else None,
                "calibration_cache": str(args.calibration_cache) if args.calibration_cache else None,
                "quantization_algorithm": (
                    "static PTQ QDQ; "
                    f"{args.calibration_method} activation calibration; "
                    "symmetric INT8 activation; per-channel INT8 weights"
                ),
                "repaired_int32_biases": repaired,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"calibration images: {len(paths)}; repaired INT32 biases: {repaired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
