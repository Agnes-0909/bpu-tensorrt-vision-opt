#!/usr/bin/env python3
"""Evaluate prediction JSON files produced by rfdetr_coco_cuda_preprocess."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(annotation: Path, prediction: Path) -> dict[str, Any]:
    coco = COCO(str(annotation))
    predictions = json.loads(prediction.read_text(encoding="utf-8"))
    if not predictions:
        raise ValueError(f"prediction file is empty: {prediction}")
    image_ids = sorted({int(item["image_id"]) for item in predictions})
    results = coco.loadRes(predictions)
    evaluator = COCOeval(coco, results, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {
        "mAP_50_95": float(evaluator.stats[0]),
        "mAP_50": float(evaluator.stats[1]),
        "mAP_75": float(evaluator.stats[2]),
        "mAP_small": float(evaluator.stats[3]),
        "mAP_medium": float(evaluator.stats[4]),
        "mAP_large": float(evaluator.stats[5]),
    }
    return {
        "prediction": str(prediction),
        "num_images": len(image_ids),
        "num_predictions": len(predictions),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, default=Path("/home/rton/annotations/instances_val2017.json"))
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    reports = [evaluate(args.annotation, path) for path in args.predictions]
    if len(reports) > 1:
        baseline = reports[0]["metrics"]
        for report in reports[1:]:
            report["delta_vs_first"] = {
                key: report["metrics"][key] - baseline[key] for key in baseline
            }
    result: Any = reports[0] if len(reports) == 1 else reports
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
