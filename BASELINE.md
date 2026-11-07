# RF-DETR XLarge TensorRT FP32 Baseline

日期：2026-10-29

## 运行环境

- GPU：NVIDIA GeForce RTX 5060 Ti，8 GB，Compute Capability 12.0
- CUDA：13.x
- TensorRT：11.4.0.106
- OpenCV：4.9.0
- Python：3.12
- COCO 评测：`pycocotools.COCOeval`

## 模型与转换

源模型为官方 RF-DETR XLarge：

```text
python_infer/output/rfdetr-xlarge.onnx
```

ONNX 输入输出：

```text
input  [1, 3, 700, 700]  FLOAT / FP32
dets   [1, 300, 4]        FLOAT / FP32
labels [1, 300, 91]      FLOAT / FP32
```

使用 TensorRT 11.4 `trtexec` 构建 FP32 baseline：

```bash
trtexec --onnx=python_infer/output/rfdetr-xlarge.onnx \
  --saveEngine=model_xlarge.engine --skipInference
```

生成的 engine 为本地文件 `model_xlarge.engine`，大小约 441.66 MiB；模型文件不进入 Git。

## COCO 评测协议

- 图片：`/home/rton/val2017/val2017`
- 检测标注：`/home/rton/annotations/instances_val2017.json`
- 图片数量：5000
- 输入预处理：BGR→RGB、torchvision 风格 bilinear resize 到 700×700、ImageNet mean/std 归一化
- 推理：持久化 TensorRT context 和预分配 CUDA buffer
- 后处理：sigmoid、每个 query 选择最高 COCO 类别、`cxcywh` 转 COCO `xywh`
- 评测阈值：`score_threshold=0.0`

COCO 指标：

| 指标 | FP32 XLarge |
|---|---:|
| mAP@[0.50:0.95] | **0.5813672** |
| AP50 | 0.7662090 |
| AP75 | 0.6324527 |
| AP small | 0.3958630 |
| AP medium | 0.6351113 |
| AP large | 0.7562676 |

完整 COCO 报告为本地生成文件 `tensort_quant/artifacts_xlarge/coco_fp32.json`。

## `tensorrt_infer` C++ 性能

运行 500 次正式迭代、50 次 warmup：

```bash
./tensorrt_infer/build/rfdetr_benchmark \
  model_xlarge.engine img/000000085772.jpg 500 50 0.5 -1
```

| 阶段 | Mean (ms) | P95 (ms) | P99 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| 前处理 | 6.8551 | 7.4863 | 8.0222 | 10.3288 |
| 推理 | 26.4743 | 27.8499 | 28.5721 | 29.0999 |
| 后处理 | 0.0711 | 0.0837 | 0.1489 | 0.2130 |

推理时间包含 H2D、TensorRT enqueue、D2H 和当前 C++ `infer()` 的每次 CUDA 分配；图像在正式循环外读取，因此不包含 JPEG 解码。

## 结果文件

- FP32 COCO 报告：`tensort_quant/artifacts_xlarge/coco_fp32.json`（本地生成）
- FP32 benchmark：`tensort_quant/artifacts_xlarge/benchmark_fp32.txt`（本地生成）
- INT8 对比报告：[tensort_quant/INT8_QUANTIZATION_REPORT.md](tensort_quant/INT8_QUANTIZATION_REPORT.md)
