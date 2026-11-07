#!/usr/bin/env python3
"""Evaluate the TensorRT RF-DETR engine on a COCO detection validation set."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def check_cuda(result: Any, expression: str) -> None:
    error = result[0] if isinstance(result, tuple) else result
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{expression} failed: {error}")


def percentile(values: list[float], probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(np.percentile(ordered, probability * 100.0, method="linear"))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("no timing samples")
    return {
        "mean_ms": float(np.mean(values)),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": float(np.max(values)),
    }


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


def preprocess(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image is None:
        raise ValueError("OpenCV returned an empty image")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = resize_torchvision_style(rgb, width, height)
    normalized = (resized - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


class TensorRTEngine:
    def __init__(self, path: Path):
        logger = trt.Logger(trt.Logger.WARNING)
        with path.open("rb") as stream:
            serialized = stream.read()
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"could not deserialize TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("could not create TensorRT execution context")

        self.input_name = None
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        if self.input_name is None:
            raise RuntimeError("engine has no input tensor")

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if len(self.input_shape) != 4 or self.input_shape[0] != 1 or self.input_shape[1] != 3:
            raise RuntimeError(f"expected input shape [1,3,H,W], got {self.input_shape}")
        self.height, self.width = self.input_shape[2], self.input_shape[3]
        self.context.set_input_shape(self.input_name, self.input_shape)
        self.output_shapes = {
            name: tuple(self.context.get_tensor_shape(name)) for name in self.output_names
        }
        self.host_outputs = {
            name: np.empty(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            for name, shape in self.output_shapes.items()
        }
        self.device_outputs = {}
        self.stream = None
        self.device_input = None
        try:
            stream_result = cudart.cudaStreamCreate()
            check_cuda(stream_result, "cudaStreamCreate")
            self.stream = stream_result[1]
            input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
            self.host_input = np.empty(self.input_shape, dtype=input_dtype)
            input_result = cudart.cudaMalloc(self.host_input.nbytes)
            check_cuda(input_result, "cudaMalloc(input)")
            self.device_input = input_result[1]
            for name, output in self.host_outputs.items():
                result = cudart.cudaMalloc(output.nbytes)
                check_cuda(result, f"cudaMalloc({name})")
                self.device_outputs[name] = result[1]
        except Exception:
            self.close()
            raise

        self.context.set_tensor_address(self.input_name, int(self.device_input))
        for name, pointer in self.device_outputs.items():
            self.context.set_tensor_address(name, int(pointer))

    def infer(self, input_array: np.ndarray) -> dict[str, np.ndarray]:
        np.copyto(self.host_input, input_array)
        check_cuda(
            cudart.cudaMemcpyAsync(
                self.device_input,
                self.host_input.ctypes.data,
                self.host_input.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync(input)",
        )
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        for name, output in self.host_outputs.items():
            check_cuda(
                cudart.cudaMemcpyAsync(
                    output.ctypes.data,
                    self.device_outputs[name],
                    output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f"cudaMemcpyAsync({name})",
            )
        check_cuda(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        return self.host_outputs

    def close(self) -> None:
        if self.device_input is not None:
            cudart.cudaFree(self.device_input)
            self.device_input = None
        for pointer in self.device_outputs.values():
            cudart.cudaFree(pointer)
        self.device_outputs.clear()
        if self.stream is not None:
            cudart.cudaStreamDestroy(self.stream)
            self.stream = None

    def __del__(self):
        self.close()


def decode_predictions(
    outputs: dict[str, np.ndarray],
    image_id: int,
    image_width: int,
    image_height: int,
    category_ids: set[int],
    score_threshold: float,
) -> list[dict[str, Any]]:
    boxes = outputs[next(name for name in outputs if "dets" in name)][0]
    labels = outputs[next(name for name in outputs if "labels" in name)][0]
    if boxes.shape[1] != 4 or labels.shape[0] != boxes.shape[0]:
        raise RuntimeError(f"unexpected output shapes: {boxes.shape}, {labels.shape}")

    allowed_slots = np.asarray(sorted(category_ids), dtype=np.int64)
    allowed_slots = allowed_slots[allowed_slots < labels.shape[1]]
    logits = np.clip(labels[:, allowed_slots], -88.0, 88.0)
    scores_all = 1.0 / (1.0 + np.exp(-logits))
    best_indices = np.argmax(scores_all, axis=1)
    scores = scores_all[np.arange(scores_all.shape[0]), best_indices]
    classes = allowed_slots[best_indices]
    keep = scores > score_threshold

    selected_boxes = boxes[keep]
    selected_scores = scores[keep]
    selected_classes = classes[keep]
    results = []
    for box, score, category_id in zip(selected_boxes, selected_scores, selected_classes):
        cx, cy, width, height = (float(value) for value in box)
        x1 = max(0.0, min(float(image_width), (cx - width / 2.0) * image_width))
        y1 = max(0.0, min(float(image_height), (cy - height / 2.0) * image_height))
        x2 = max(0.0, min(float(image_width), (cx + width / 2.0) * image_width))
        y2 = max(0.0, min(float(image_height), (cy + height / 2.0) * image_height))
        results.append({
            "image_id": image_id,
            "category_id": int(category_id),
            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
            "score": float(score),
        })
    return results


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="COCO mAP evaluation for RF-DETR TensorRT engine")
    parser.add_argument("--engine", type=Path, default=root / "model.engine")
    parser.add_argument("--images", type=Path, default=Path("/home/rton/val2017/val2017"))
    parser.add_argument("--annotation", type=Path, default=Path("/home/rton/annotations/instances_val2017.json"))
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=root / "coco_predictions.json")
    parser.add_argument("--report", type=Path, default=root / "coco_baseline.json")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images; 0 means all")
    args = parser.parse_args()

    if not args.annotation.exists():
        raise FileNotFoundError(
            f"COCO detection annotation not found: {args.annotation}. "
            "The captions_val2017.json file is not a detection annotation."
        )
    coco = COCO(str(args.annotation))
    image_ids = coco.getImgIds()
    if args.limit > 0:
        image_ids = image_ids[: args.limit]
    category_ids = set(coco.getCatIds())
    image_info = {int(item["id"]): item for item in coco.loadImgs(image_ids)}

    engine = TensorRTEngine(args.engine)
    predictions: list[dict[str, Any]] = []
    preprocess_times: list[float] = []
    inference_times: list[float] = []
    postprocess_times: list[float] = []
    missing = []
    total_detections = 0
    wall_begin = time.perf_counter()
    try:
        for index, image_id in enumerate(image_ids, start=1):
            info = image_info[image_id]
            image_path = args.images / info["file_name"]
            if not image_path.exists():
                missing.append(str(image_path))
                continue

            preprocess_begin = time.perf_counter()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            input_array = preprocess(image, engine.width, engine.height)
            preprocess_times.append((time.perf_counter() - preprocess_begin) * 1000.0)

            inference_begin = time.perf_counter()
            outputs = engine.infer(input_array)
            inference_times.append((time.perf_counter() - inference_begin) * 1000.0)

            postprocess_begin = time.perf_counter()
            image_predictions = decode_predictions(
                outputs,
                image_id,
                int(info["width"]),
                int(info["height"]),
                category_ids,
                args.score_threshold,
            )
            postprocess_times.append((time.perf_counter() - postprocess_begin) * 1000.0)
            predictions.extend(image_predictions)
            total_detections += len(image_predictions)
            if index % 250 == 0 or index == len(image_ids):
                print(f"processed {index}/{len(image_ids)} images", flush=True)
    finally:
        engine.close()

    if missing:
        raise RuntimeError(f"missing {len(missing)} images; first missing file: {missing[0]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions), encoding="utf-8")

    coco_results = coco.loadRes(predictions)
    evaluator = COCOeval(coco, coco_results, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    report = {
        "engine": str(args.engine),
        "images": str(args.images),
        "annotation": str(args.annotation),
        "num_images": len(image_ids),
        "num_predictions": len(predictions),
        "average_detections_per_image": total_detections / len(image_ids),
        "score_threshold": args.score_threshold,
        "metrics": {
            "mAP_50_95": float(evaluator.stats[0]),
            "mAP_50": float(evaluator.stats[1]),
            "mAP_75": float(evaluator.stats[2]),
            "mAP_small": float(evaluator.stats[3]),
            "mAP_medium": float(evaluator.stats[4]),
            "mAP_large": float(evaluator.stats[5]),
        },
        "timing_ms": {
            "preprocess": summarize(preprocess_times),
            "inference": summarize(inference_times),
            "postprocess": summarize(postprocess_times),
            "wall_total_ms": (time.perf_counter() - wall_begin) * 1000.0,
        },
        "notes": [
            "Image decoding is included in preprocess timing.",
            "Inference uses one persistent TensorRT context and preallocated CUDA buffers.",
            "For the current sparse COCO output layout, category slot equals COCO category_id and slot 0 is excluded.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"predictions saved to {args.output}")
    print(f"report saved to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
