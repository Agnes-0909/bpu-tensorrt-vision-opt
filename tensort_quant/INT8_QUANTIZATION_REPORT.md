# RF-DETR XLarge INT8 TensorRT 量化报告

## 1. 实验对象与转换

本次实验统一使用 RF-DETR XLarge，而不是 Nano。官方 XLarge 权重通过 `rfdetr[plus]` 下载到 `/home/rton/.roboflow/models/rf-detr-xlarge.pth`，再使用 RF-DETR 官方导出接口转换为 ONNX：

```python
from rfdetr import RFDETRXLarge
model = RFDETRXLarge(num_classes=90)
model.export(format="onnx", opset_version=16, output_dir="python_infer/output", output_name="rfdetr-xlarge.onnx", fp16=False)
```

产物为本地文件 `python_infer/output/rfdetr-xlarge.onnx`，输入固定为 `[1, 3, 700, 700]`，输出为 `dets [1,300,4]` 和 `labels [1,300,91]`。FP32 TensorRT baseline 为本地文件 `model_xlarge.engine`。ONNX 和 engine 不进入 Git；TensorRT engine 使用 TensorRT 11.4 `trtexec` 从 ONNX 构建，未启用显式 FP16，因此 baseline 按 FP32 engine 统计。

## 2. 校准数据与量化策略

采用静态后训练量化（PTQ），校准数据为 COCO val2017：

- 图片目录：`/home/rton/val2017/val2017`；
- 按文件名排序后均匀采样 64 张图片；
- 预处理与评测一致：BGR→RGB、bilinear resize 到 700×700（antialias=False）、ImageNet mean/std 归一化；
- MinMax 激活校准；
- INT8 激活和 INT8 权重；
- 对称量化，权重 per-channel；
- 输出为显式 Q/DQ ONNX。

### 2.1 完整计算图与本报告方案的实际范围

完整图先分为四类：DINOv2 encoder（patch embedding、12 个 Transformer block 和 FFN）、encoder 输出筛选分支、DETR decoder（5 个 layer，每层包含 self-attention、deformable cross-attention 和 FFN）、以及 class/bbox detection head。encoder 和 decoder 中的 attention 还要区分 Q/K/V projection、QKᵀ、Softmax、AV、output projection；这些不是同一个量化对象。

| 方案 | 实际插入 INT8 Q/DQ 的层 | 明确保留 FP32/原精度的层 |
|---|---|---|
| 全量 Conv+MatMul MinMax | 所有 Conv；encoder 的 Q/K/V、QKᵀ、AV、attention output、FFN `fc1/fc2`；decoder 的 self/cross-attention MatMul、FFN、reference point 和检测 head MatMul | LayerNorm、residual Add、Softmax、Reshape、Gather、GridSample 等非 Conv/MatMul 节点自身未量化，但会接收上游量化误差 |
| Conv-only MinMax | backbone/projector 的 9 个 Conv | 所有 encoder/decoder MatMul，包括 attention projection、QKᵀ、AV、FFN、reference point、class/bbox head；LayerNorm、residual、Softmax、GridSample 也保持原精度 |
| 当前 Conv-only 混合部署候选 | 与 Conv-only MinMax 相同：只量化 9 个 Conv | DINOv2/Transformer MatMul、decoder、LayerNorm、residual、Softmax、reference point 和 detection head 全部保护 |

因此，“Conv-only”不是“整个 encoder 量化”，而是只量化 9 个 Conv；报告中后续提到的 encoder SmoothQuant 属于另一条实验线，量化的是 encoder Linear 的输入/权重，不能与这里的 Conv-only 结果混为一谈。

实验中发现 XLarge 的 DINOv2/Transformer 编码器对 MatMul 激活量化非常敏感。全量 Conv+MatMul INT8 的敏感性实验结果如下：TensorRT 推理均值从 26.4743 ms 降至 11.3143 ms，减少 15.1600 ms（57.26%，约 2.34× 加速）；包含预处理、推理和后处理的三阶段均值从 33.4005 ms 降至 18.0970 ms，减少 45.82%。但 COCO mAP@[0.50:0.95] 从 0.58137 降至 0.12592，绝对下降 0.45545，即 45.54 个百分点（相对下降 78.34%），不能作为部署方案。因此最终采用保精度的混合策略：

- **INT8**：主干中的 Conv；
- **FP32**：DINOv2/Transformer MatMul、残差加法、归一化、softmax、分类头和框回归头。

最终产物保存在本地目录 `artifacts_xlarge_conv/`，包括 ONNX、engine 和 calibration manifest；这些生成文件不进入 Git。TensorRT 不接受 ONNX Runtime 产生的 INT32 bias `DequantizeLinear` 输入，量化脚本会将这类 bias 折叠为数值等价的 FP32 常量。

## 3. 模型体积

| 项目 | FP32 | INT8 Conv-only | 变化 |
|---|---:|---:|---:|
| TensorRT engine | 441.66 MiB | 420.60 MiB | 减少 4.77% |

XLarge 的主体计算是 Transformer MatMul，最终策略只量化 9 个 Conv，因此 engine 体积收益有限；该选择主要用于控制精度损失。

## 4. `tensorrt_infer` C++ 性能

单张 `img/000000085772.jpg`，预热 50 次，正式迭代 500 次。推理时间包含 H2D、TensorRT enqueue、D2H 和当前 C++ 实现的每次 CUDA 分配。

| 阶段 | FP32 均值 | INT8 均值 | 变化 |
|---|---:|---:|---:|
| 预处理 | 6.8551 ms | 6.7192 ms | -1.98% |
| TensorRT 推理 | 26.4743 ms | 25.7103 ms | **-2.89%** |
| 后处理 | 0.0711 ms | 0.0722 ms | +1.55% |
| 三阶段合计 | 33.4005 ms | 32.5017 ms | **-2.69%** |

推理 P95 从 27.8499 ms 降至 26.2151 ms。机器可读结果保存在本地 `artifacts_xlarge_conv/comparison.json`。

## 5. COCO val2017 精度

使用 `/home/rton/annotations/instances_val2017.json`，完整 5000 张图片，`score_threshold=0.0`，FP32/INT8 使用相同的评测脚本和后处理。

| 指标 | FP32 | INT8 Conv-only | 绝对变化 |
|---|---:|---:|---:|
| mAP@[0.50:0.95] | 0.58137 | 0.58004 | **-0.00133** |
| mAP@0.50 | 0.76621 | 0.76493 | -0.00128 |
| mAP@0.75 | 0.63245 | 0.63024 | -0.00221 |
| mAP small | 0.39586 | 0.39432 | -0.00154 |
| mAP medium | 0.63511 | 0.63233 | -0.00278 |
| mAP large | 0.75627 | 0.75716 | +0.00090 |

主指标 mAP@[0.50:0.95] 绝对损失仅 0.00133，相对下降约 0.23%。FP32/INT8 COCO 报告和对比汇总保存在本地 `artifacts_xlarge*/` 目录。

## 6. 结论

在 RTX 5060 Ti、TensorRT 11.4 上，XLarge Conv-only INT8 混合精度方案将 TensorRT 推理均值降低约 2.89%，三阶段端到端耗时降低约 2.69%，同时将 COCO mAP 损失控制在 0.00133。由于 XLarge 的主要计算量来自 Transformer MatMul，而这些层为保证精度保留 FP32，所以速度和模型体积收益低于 Nano 或全量 MatMul INT8 方案。

全量 Conv+MatMul INT8 的结果保存在 `artifacts_xlarge/`，仅作为敏感性实验，不建议用于部署。

全量量化的逐项结果：

| 指标 | FP32 | 全量 Conv+MatMul INT8 | 绝对变化 |
|---|---:|---:|---:|
| TensorRT 推理均值 | 26.4743 ms | 11.3143 ms | -15.1600 ms（-57.26%） |
| 三阶段合计均值 | 33.4005 ms | 18.0970 ms | -15.3035 ms（-45.82%） |
| mAP@[0.50:0.95] | 0.58137 | 0.12592 | -0.45545（-45.54 个百分点） |
| mAP@0.50 | 0.76621 | 0.18841 | -0.57779（-57.78 个百分点） |
