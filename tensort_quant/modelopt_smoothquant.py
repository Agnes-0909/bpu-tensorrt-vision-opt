#!/usr/bin/env python3
"""Export a selective RF-DETR XLarge SmoothQuant PTQ ONNX model.

Only the DINOv2 encoder's Linear layers are quantized. Decoder attention,
LayerNorm, residual paths, deformable attention and detection heads stay in
their original precision because the all-module SmoothQuant trial was not
accurate enough for this detector.
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


def resize_torchvision_style(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
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


def preprocess(path: Path, width: int = 700, height: int = 700) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read calibration image: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = resize_torchvision_style(rgb, width, height)
    normalized = (resized - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32))


def select_images(directory: Path, count: int) -> list[Path]:
    paths = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"no JPEG images found in {directory}")
    indices = np.linspace(0, len(paths) - 1, min(count, len(paths)), dtype=np.int64)
    return [paths[int(index)] for index in indices]


def smoothquant_config() -> dict:
    # ModelOpt applies the last matching entry. Disable every quantizer first,
    # then enable only DINOv2 encoder Linear weights and inputs.
    dino_linear = "*backbone.0.encoder.encoder.encoder.layer.*.*"
    quant_cfg = [
        *_base_disable_all,
        {
            "quantizer_name": dino_linear + "weight_quantizer",
            "parent_class": "nn.Linear",
            "cfg": {"num_bits": 8, "axis": 0},
        },
        {
            "quantizer_name": dino_linear + "input_quantizer",
            "parent_class": "nn.Linear",
            "cfg": {"num_bits": 8, "axis": None},
        },
        *_default_disabled_quantizer_cfg,
    ]
    return {"quant_cfg": quant_cfg, "algorithm": "smoothquant"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("/home/rton/val2017/val2017"))
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tensort_quant/artifacts_xlarge_smoothquant")
    )
    parser.add_argument("--output-name", default="rfdetr-xlarge-smoothquant")
    args = parser.parse_args()
    if args.calibration_count <= 0:
        raise ValueError("--calibration-count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = select_images(args.images, args.calibration_count)

    # ModelOpt 0.47 emits a compatibility warning for the local Transformers
    # version, but RF-DETR's own modules are quantized successfully.
    warnings.filterwarnings("ignore", message="transformers .* is not tested")
    detector = RFDETRXLarge(num_classes=90)
    model = detector.model.model.cuda().eval()
    calibration = [preprocess(path).cuda() for path in paths]

    def forward_loop(calibrated_model: torch.nn.Module) -> None:
        with torch.no_grad():
            for sample in calibration:
                calibrated_model(sample)

    model = mtq.quantize(model, smoothquant_config(), forward_loop)
    detector.model.model = model
    output = detector.export(
        output_dir=str(args.output_dir),
        output_name=args.output_name,
        format="onnx",
        opset_version=16,
        fp16=False,
        verbose=False,
    )
    (args.output_dir / "calibration_manifest.json").write_text(
        json.dumps(
            {
                "source_model": "python_infer/output/rfdetr-xlarge.onnx",
                "quantized_model": str(output),
                "calibration_images": [str(path) for path in paths],
                "calibration_count": len(paths),
                "algorithm": "SmoothQuant",
                "quantized_scope": "DINOv2 encoder Linear layers only",
                "precision_exclusions": [
                    "Conv",
                    "decoder",
                    "LayerNorm",
                    "residuals",
                    "softmax",
                    "deformable attention",
                    "classification and box heads",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"exported {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
