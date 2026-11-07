#!/usr/bin/env python3
"""Export encoder-SmoothQuant plus a selected decoder Linear scope.

The encoder selection is kept identical to the completed SQ256 experiment.
Only explicitly selected decoder ``nn.Linear`` modules are enabled in the
ModelOpt SmoothQuant configuration; attention softmax, residuals, deformable
sampling, packed self-attention projections and heads remain FP32.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from modelopt.torch.quantization.config import _base_disable_all, _default_disabled_quantizer_cfg
from rfdetr import RFDETRXLarge
import modelopt.torch.quantization as mtq

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def resize_torchvision_style(rgb: np.ndarray, width: int = 700, height: int = 700) -> np.ndarray:
    source_height, source_width = rgb.shape[:2]
    ys = (np.arange(height, dtype=np.float32) + 0.5) * source_height / height - 0.5
    xs = (np.arange(width, dtype=np.float32) + 0.5) * source_width / width - 0.5
    y_floor, x_floor = np.floor(ys), np.floor(xs)
    y0 = np.clip(y_floor.astype(np.int32), 0, source_height - 1)
    y1 = np.clip(y0 + 1, 0, source_height - 1)
    x0 = np.clip(x_floor.astype(np.int32), 0, source_width - 1)
    x1 = np.clip(x0 + 1, 0, source_width - 1)
    wy = np.clip(ys - y_floor, 0.0, 1.0).reshape(-1, 1, 1)
    wx = np.clip(xs - x_floor, 0.0, 1.0).reshape(1, -1, 1)
    top = rgb[y0[:, None], x0[None, :]] * (1 - wx) + rgb[y0[:, None], x1[None, :]] * wx
    bottom = rgb[y1[:, None], x0[None, :]] * (1 - wx) + rgb[y1[:, None], x1[None, :]] * wx
    return top * (1 - wy) + bottom * wy


def preprocess(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read calibration image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (resize_torchvision_style(rgb) - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32))


def select_images(directory: Path, count: int) -> list[Path]:
    paths = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"no JPEG images found in {directory}")
    indices = np.linspace(0, len(paths) - 1, min(count, len(paths)), dtype=np.int64)
    return [paths[int(i)] for i in indices]


def decoder_patterns(scope: str) -> list[str]:
    all_linear = [
        "*transformer.decoder.layers.*.cross_attn.value_proj",
        "*transformer.decoder.layers.*.cross_attn.output_proj",
        "*transformer.decoder.layers.*.cross_attn.attention_weights",
        "*transformer.decoder.layers.*.cross_attn.sampling_offsets",
        "*transformer.decoder.layers.*.linear1",
        "*transformer.decoder.layers.*.linear2",
    ]
    scopes = {
        "all_linear": all_linear,
        "ffn": ["*transformer.decoder.layers.*.linear1", "*transformer.decoder.layers.*.linear2"],
        "cross_value_output": [
            "*transformer.decoder.layers.*.cross_attn.value_proj",
            "*transformer.decoder.layers.*.cross_attn.output_proj",
        ],
        "cross_sampling_offsets": ["*transformer.decoder.layers.*.cross_attn.sampling_offsets"],
        "cross_attention_weights": ["*transformer.decoder.layers.*.cross_attn.attention_weights"],
    }
    if scope not in scopes:
        raise ValueError(f"unknown scope {scope}: {', '.join(scopes)}")
    return scopes[scope]


def smoothquant_config(scope: str) -> dict:
    # Keep the encoder selection exactly aligned with the completed SQ256 run.
    entries = [
        {
            "quantizer_name": "*backbone.0.encoder.encoder.encoder.layer.*.*weight_quantizer",
            "parent_class": "nn.Linear",
            "cfg": {"num_bits": 8, "axis": 0},
        },
        {
            "quantizer_name": "*backbone.0.encoder.encoder.encoder.layer.*.*input_quantizer",
            "parent_class": "nn.Linear",
            "cfg": {"num_bits": 8, "axis": None},
        },
    ]
    for module in decoder_patterns(scope):
        entries.extend([
            {"quantizer_name": module + ".weight_quantizer", "parent_class": "nn.Linear", "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": module + ".input_quantizer", "parent_class": "nn.Linear", "cfg": {"num_bits": 8, "axis": None}},
        ])
    return {"quant_cfg": [*_base_disable_all, *entries, *_default_disabled_quantizer_cfg], "algorithm": "smoothquant"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["all_linear", "ffn", "cross_value_output", "cross_sampling_offsets", "cross_attention_weights"], default="all_linear")
    parser.add_argument("--images", type=Path, default=Path("/home/rton/val2017/val2017"))
    parser.add_argument("--calibration-count", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("tensorrt_quant_decoder/artifacts/smoothquant_all_linear_256"))
    parser.add_argument("--output-name", default="rfdetr-xlarge-decoder-smoothquant-all-linear-256")
    args = parser.parse_args()
    if args.calibration_count <= 0:
        raise ValueError("--calibration-count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = select_images(args.images, args.calibration_count)
    warnings.filterwarnings("ignore", message="transformers .* is not tested")
    detector = RFDETRXLarge(num_classes=90)
    model = detector.model.model.cuda().eval()
    calibration = [preprocess(path).cuda() for path in paths]

    def forward_loop(calibrated_model: torch.nn.Module) -> None:
        with torch.no_grad():
            for sample in calibration:
                calibrated_model(sample)

    model = mtq.quantize(model, smoothquant_config(args.scope), forward_loop)
    detector.model.model = model
    output = detector.export(output_dir=str(args.output_dir), output_name=args.output_name, format="onnx", opset_version=16, fp16=False, verbose=False)
    manifest = {
        "source_model": "python_infer/output/rfdetr-xlarge.onnx",
        "quantized_model": str(output),
        "calibration_images": [str(path) for path in paths],
        "calibration_count": len(paths),
        "algorithm": "SmoothQuant",
        "quantized_scope": "DINOv2 encoder Linear layers plus decoder " + args.scope,
        "decoder_module_patterns": decoder_patterns(args.scope),
        "precision_exclusions": ["Conv", "self-attention packed projections", "LayerNorm", "residuals", "softmax", "deformable sampling", "classification and box heads"],
    }
    (args.output_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"exported {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
