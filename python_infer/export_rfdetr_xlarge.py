#!/usr/bin/env python3
"""Export the official RF-DETR XLarge checkpoint to a static FP32 ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr import RFDETRXLarge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'output')
    parser.add_argument('--output-name', default='rfdetr-xlarge.onnx')
    parser.add_argument('--opset-version', type=int, default=16)
    parser.add_argument('--num-classes', type=int, default=90)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = RFDETRXLarge(num_classes=args.num_classes)
    model.export(
        format='onnx',
        opset_version=args.opset_version,
        output_dir=str(args.output_dir),
        output_name=args.output_name,
        fp16=False,
    )
    print(args.output_dir / args.output_name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
