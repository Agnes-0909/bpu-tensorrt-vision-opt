# 当前阶段部署候选

## 方案

```text
Encoder: ModelOpt SmoothQuant，256 张 COCO 校准图
Decoder: ModelOpt SmoothQuant，all_linear，256 张 COCO 校准图
Head: 保持 FP32/原始精度
```

本方案对应 `smoothquant_decoder.py --scope all_linear --calibration-count 256`，导出的 ONNX 和 TensorRT engine 位于本地：

```text
tensorrt_quant_decoder/artifacts/smoothquant_all_linear_256/
```

## 完整 COCO 结果

| 项目 | 结果 |
|---|---:|
| Encoder SQ256 + Decoder FP32 baseline mAP | 0.580251 |
| Encoder SQ256 + Decoder SQ all Linear mAP | **0.580280** |
| mAP 变化 | **+0.000029** |
| 应用级 TensorRT 推理均值 | 18.710 ms |
| baseline 应用级 TensorRT 推理均值 | 18.259 ms |
| TensorRT engine 大小 | 163.8 MiB |

decoder SmoothQuant 当前优先作为精度保持方案。它没有显示出稳定的端到端加速，因此若目标是最低延迟，应继续使用 TensorRT profile 对 decoder plugin、Q/DQ 边界和 kernel tactic 做专项优化。

说明：D7 `attention_weights` 的 MinMax 单组实验 mAP 为 0.580713，是单项 mAP 最高的窄范围候选；这里选择 all-linear SmoothQuant，是因为它覆盖范围更完整、与 encoder 使用同一类平滑方法，且相对固定 baseline 的变化只有 +0.000029，更适合作为当前阶段的组合部署候选。

## 复现

```bash
conda activate torch
python smoothquant_decoder.py --scope all_linear --calibration-count 256

conda activate TensorRT
python ../tensort_quant/build_int8_engine.py \
  --onnx artifacts/smoothquant_all_linear_256/rfdetr-xlarge-decoder-smoothquant-all-linear-256.onnx \
  --engine artifacts/smoothquant_all_linear_256/rfdetr-xlarge-decoder-smoothquant-all-linear-256.engine
```
