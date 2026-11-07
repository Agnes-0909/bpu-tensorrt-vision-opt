# RF-DETR XLarge PTQ 分组实验记录

本文记录不涉及 QAT 的 TensorRT PTQ 实验。分组 Q/DQ 候选使用相同的 RF-DETR XLarge ONNX、静态输入 `[1,3,700,700]`、64 张 COCO val2017 校准图片、显式 Q/DQ、INT8 对称激活和 per-channel 权重；SmoothQuant 另外比较了 64/256 张校准图片。所有 engine 均使用 TensorRT 11.4 在 RTX 5060 Ti 上构建。

## 1. 实验脚本与复现入口

本报告由量化、engine 构建、精度评测和 benchmark 的结果汇总而成，没有单独的脚本自动生成整个 Markdown 报告。各脚本的职责如下：

| 脚本 | 作用 |
|---|---|
| [`quantize_int8.py`](quantize_int8.py) | 主要的 ONNX Runtime 静态 PTQ 脚本，生成显式 Q/DQ ONNX；通过 `--quantize-ops`、`--include-pattern` 和 `--exclude-pattern` 实现 Decoder、DINOv2 FFN、QKV、QKVO 等分组。 |
| [`modelopt_smoothquant.py`](modelopt_smoothquant.py) | 使用 ModelOpt 导出选择性 SmoothQuant ONNX，仅量化 DINOv2 encoder 的 Linear；分别以 64/256 张图片校准。 |
| [`build_int8_engine.py`](build_int8_engine.py) | 将各分组的 Q/DQ ONNX 构建为 TensorRT INT8 engine。 |
| [`../evaluate_coco_tensorrt.py`](../evaluate_coco_tensorrt.py) | 使用相同的 COCO 后处理和 `score_threshold=0.0` 完成 FP32/INT8 COCO val2017 评测。 |
| `tensorrt_infer/build/rfdetr_benchmark` | 对单张图片执行预热、迭代 benchmark，并记录均值和 P95。 |
| [`run_pipeline.sh`](run_pipeline.sh) | 标准 Conv-only PTQ 的端到端示例入口；报告中的全部分组实验还需要分别调用上述量化脚本。 |

其中，`quantize_int8.py` 是本报告大部分 Q/DQ 分组实验的主要脚本，SmoothQuant 部分则由 `modelopt_smoothquant.py` 完成。

## 2. 实验分组

| 产物目录 | INT8 节点 | 保留 FP32 的主要部分 |
|---|---|---|
| `artifacts_xlarge_decoder/` | Conv + DETR decoder MatMul | DINOv2 backbone、归一化、残差、softmax、检测头 |
| `artifacts_xlarge_dino_ffn/` | DINOv2 MLP `fc1/fc2` MatMul | DINOv2 attention、DETR decoder、归一化、残差、softmax、检测头 |
| `artifacts_xlarge_dino_qkv/` | DINOv2 attention 的 query/key/value projection MatMul | DINOv2 attention score/output、MLP、DETR decoder、归一化、残差、softmax、检测头 |
| `artifacts_xlarge_dino_qkvo/` | DINOv2 attention 的 Q/K/V + output projection MatMul | attention score、Softmax、LayerNorm、MLP、DETR decoder、检测头 |
| `artifacts_xlarge_smoothquant/` | ModelOpt SmoothQuant：DINOv2 encoder Linear | Conv、decoder、LayerNorm、残差、softmax、deformable attention、检测头 |
| `artifacts_xlarge_smoothquant_256/` | 同上，256 张校准图片 | 同上 |

`artifacts_xlarge_decoder/` 使用已有产物重新完成了完整 5000 张 COCO 评测；其原先只有 100 张图片的结果不用于结论。

## 3. COCO val2017 精度

**范围说明：本文中的“全量 INT8”与后续 encoder 阶段报告中的“MinMax”不是同一个实验。** 本文表格中的 `artifacts_xlarge/` 是 Conv 和所有 MatMul（包括 DINOv2 的 attention score `QKᵀ`、attention value `AV`、encoder projection、decoder MatMul 等）一起做 INT8；后续 `error_analysis_encoder_compare/` 的 MinMax 只量化 DINOv2 encoder 中每个 block 的 Q/K/V、attention output、FFN fc1/fc2 共 72 个 projection MatMul，attention score、Softmax、LayerNorm、残差、decoder 和检测头保持 FP32。因此不能只按算法名称比较两者的逐层 relative L1 或 mAP。

评测使用 `score_threshold=0.0`，与 FP32 baseline 使用相同的评测脚本和后处理。精度变化以百分点表示。

| 方案 | mAP@[0.50:0.95] | 相对 FP32 | mAP@0.50 | mAP small | mAP medium | mAP large |
|---|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 0.58137 | — | 0.76621 | 0.39586 | 0.63511 | 0.75627 |
| Decoder-only INT8 | 0.57676 | **-0.46 个百分点** | 0.76290 | 0.38969 | 0.63172 | 0.75389 |
| DINOv2 FFN INT8 | 0.53031 | **-5.11 个百分点** | 0.70323 | 0.34373 | 0.57837 | 0.70681 |
| DINOv2 QKV projection INT8 | 0.58146 | **+0.01 个百分点** | 0.76640 | 0.39488 | 0.63461 | 0.76035 |
| DINOv2 QKVO projection INT8 | 0.58131 | **-0.005 个百分点** | 0.76429 | 0.39509 | 0.63581 | 0.75616 |
| DINOv2 encoder SmoothQuant INT8 | 0.57930 | **-0.21 个百分点** | 0.76348 | 0.39490 | 0.63116 | 0.75639 |
| DINOv2 encoder SmoothQuant INT8（256 张校准） | 0.58025 | **-0.11 个百分点** | 0.76465 | 0.39359 | 0.63434 | 0.75606 |

结论：当前校准方法下，DINOv2 Q/K/V 以及 attention output projection 都可以量化而几乎不损失精度；QKVO 方案比 QKV-only 多量化一个 projection，完整 COCO mAP 仍只下降 0.005 个百分点。DINOv2 FFN 对 INT8 明显敏感；decoder-only 方案精度可接受但主指标仍损失约 0.46 个百分点。

SmoothQuant 结果来自 ModelOpt 0.47 的 PTQ 导出：只对 DINOv2 encoder 的 Linear 输入和权重做 SmoothQuant INT8，decoder、LayerNorm、残差、softmax、deformable attention 和检测头保持原精度。完整 COCO 评测的精度损失为 0.20655 个百分点；相较于单独量化 QKV，速度收益更高。

## 4. 单图 TensorRT benchmark

单张 `img/000000085772.jpg`，预热 50 次、正式迭代 500 次。推理均值包含 H2D、TensorRT enqueue、D2H 和当前 C++ 实现的 CUDA 分配。

| 方案 | TensorRT 推理均值 | 相对 FP32 | P95 | engine |
|---|---:|---:|---:|---:|
| FP32 baseline | 26.4743 ms | — | 27.8499 ms | 441.66 MiB |
| Decoder-only INT8 | 25.7537 ms | **-2.72%** | 26.7470 ms | 388.45 MiB |
| DINOv2 FFN INT8 | 20.2178 ms | **-23.63%** | 21.3426 ms | 280.29 MiB |
| DINOv2 QKV projection INT8 | 24.4891 ms | **-7.50%** | 25.2922 ms | 381.72 MiB |
| DINOv2 QKVO projection INT8 | 23.4284 ms | **-11.51%** | 24.1547 ms | 362.25 MiB |
| DINOv2 encoder SmoothQuant INT8 | 18.0908 ms | **-31.67%** | 18.9724 ms | 202.59 MiB |
| DINOv2 encoder SmoothQuant INT8（256 张校准） | 18.8348 ms | **-28.86%** | 19.8147 ms | 202.71 MiB |

这些结果均为单独运行的 benchmark。之前两个分组同时运行时 GPU 互相争用，产生的较高延迟值已废弃。
SmoothQuant engine 相对 FP32 的体积减少 **54.13%**，推理约 **1.46×** 加速。

将校准集从 64 张扩大到 256 张后，完整 COCO mAP 从 0.57930 提升到 0.58025，精度损失从 0.21 个百分点降到 **0.11 个百分点**；engine 体积基本不变。两次单图 C++ benchmark 受到 GPU 时钟和系统负载影响，256 张版本测得 18.83 ms，部署选型应以同一时段重复测量为准。

## 5. SmoothQuant 失败对照

全模块 SmoothQuant（包括 decoder 和检测头）只在 100 张图片上得到 mAP 0.09713，属于明显失真，因此没有继续跑完整验证集，也不作为候选方案。选择性 SmoothQuant 使用 64 张校准图后恢复到完整 COCO mAP 0.57930；这说明量化范围和校准覆盖比“全模型统一量化”更关键。

逐层 Polygraphy 分析也验证了这个范围差异：全量 Conv+MatMul MinMax 在第 0 个 encoder block 的 attention score `MatMul_1` 就达到 relative L1=0.663，后续 score MatMul 最高达到 2.491；这类节点不在 encoder-only MinMax 的 72 个量化节点中。

此前选择性 SmoothQuant 的单图校准试验 mAP 只有 0.44336，同样被视为校准不足的预实验，不用于最终结论。

## 6. 下一步

- 扩大校准集到 256/512 张，并比较 MinMax、percentile/entropy 和 SmoothQuant；
- 对 DINOv2 QKV 继续加入 attention output projection，保持 QKᵀ、Softmax 和 LayerNorm 为 FP16/FP32；
- 评估 FP8 Transformer PTQ；
- 继续保留 QAT 相关工作，不在本阶段验证。
