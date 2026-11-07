#!/usr/bin/env python3
"""Build an INT8 TensorRT engine from an explicit Q/DQ ONNX model."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=root / "artifacts_xlarge_conv/rfdetr-xlarge-conv-int8.onnx")
    parser.add_argument("--engine", type=Path, default=root / "artifacts_xlarge_conv/rfdetr-xlarge-conv-int8.engine")
    parser.add_argument(
        "--trtexec",
        type=Path,
        default=Path(os.environ.get("TRTEXEC", "/home/rton/TensorRT/TensorRT-11.4.0.106/bin/trtexec")),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.onnx.exists():
        raise FileNotFoundError(args.onnx)
    if not args.trtexec.exists():
        raise FileNotFoundError(args.trtexec)
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.trtexec),
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        "--skipInference",
    ]
    if args.verbose:
        command.append("--verbose")
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"wrote {args.engine} ({args.engine.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
