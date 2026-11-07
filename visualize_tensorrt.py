#!/usr/bin/env python3
"""Run the TensorRT executable and render its detections with OpenCV."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import cv2


DETECTION_RE = re.compile(
    r"^\d+: class=(\d+) score=([-+0-9.eE]+) "
    r"box=\[([-+0-9.eE]+),\s+([-+0-9.eE]+),\s+([-+0-9.eE]+),\s+([-+0-9.eE]+)\]$"
)


def parse_detections(stdout: str):
    detections = []
    for line in stdout.splitlines():
        match = DETECTION_RE.match(line.strip())
        if match is None:
            continue
        class_id, score, x1, y1, x2, y2 = match.groups()
        detections.append((int(class_id), float(score), *(float(v) for v in (x1, y1, x2, y2))))
    return detections


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Render RF-DETR TensorRT detections with OpenCV")
    parser.add_argument("--binary", type=Path, default=root / "tensorrt_infer/build/rfdetr_trt")
    parser.add_argument("--engine", type=Path, default=root / "model.engine")
    parser.add_argument("--image", type=Path, default=root / "img/000000085772.jpg")
    parser.add_argument("--output", type=Path, default=root / "rfdetr_result.jpg")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--background-class", type=int, default=-1)
    args = parser.parse_args()

    command = [
        str(args.binary),
        str(args.engine),
        str(args.image),
        str(args.threshold),
        str(args.background_class),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    detections = parse_detections(result.stdout)
    rendered = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if rendered is None:
        raise RuntimeError(f"OpenCV could not read image: {args.image}")

    thickness = max(1, round(min(rendered.shape[:2]) / 500))
    font_scale = max(0.45, min(rendered.shape[:2]) / 1200)
    for class_id, score, x1, y1, x2, y2 in detections:
        left = max(0, min(rendered.shape[1] - 1, round(x1)))
        top = max(0, min(rendered.shape[0] - 1, round(y1)))
        right = max(0, min(rendered.shape[1] - 1, round(x2)))
        bottom = max(0, min(rendered.shape[0] - 1, round(y2)))
        cv2.rectangle(rendered, (left, top), (right, bottom), (0, 255, 0), thickness)

        label = f"class {class_id}: {score:.3f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_top = max(text_height + baseline, top)
        cv2.rectangle(
            rendered,
            (left, label_top - text_height - baseline),
            (min(rendered.shape[1] - 1, left + text_width), label_top),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            rendered,
            label,
            (left, label_top - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rendered):
        raise RuntimeError(f"OpenCV could not write output image: {args.output}")
    print(f"saved {args.output} ({len(detections)} detections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
