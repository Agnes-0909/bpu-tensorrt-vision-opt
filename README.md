# RF-DETR XLarge TensorRT 量化实验

本仓库记录 RF-DETR XLarge 从 PyTorch 导出、FP32 TensorRT 基线、计算图分析，到 encoder 和 decoder 选择性 INT8 量化的完整过程。模型文件、TensorRT engine、COCO 逐图预测和大体积实验产物保留在本地，但默认不会进入 Git。

## 当前方案

当前阶段推荐方案是：

```text
PyTorch RF-DETR XLarge
  → FP32 ONNX
  → encoder SmoothQuant（256 张 COCO 校准图）
  → decoder Linear SmoothQuant（同一组 256 张校准图）
  → TensorRT engine
```

对应本地模型和 engine：

```text
tensorrt_quant_decoder/artifacts/smoothquant_all_linear_256/
```

完整 COCO 结果：mAP@[0.50:0.95] = **0.580280**。固定 encoder SmoothQuant、decoder FP32 基线为 0.580251，因此 decoder SmoothQuant 的精度变化约为 **+0.000029**。当前应用级 TensorRT 推理均值为 18.710 ms；decoder 量化的主要收益是模型权重压缩，速度收益目前不稳定。

## 实验逻辑

### 1. PyTorch → FP32 ONNX

```bash
conda activate torch
python python_infer/export_rfdetr_xlarge.py
```

`python_infer/export_rfdetr_xlarge.py` 使用 RF-DETR 官方导出接口生成 `python_infer/output/rfdetr-xlarge.onnx`。输入固定为 `1×3×700×700`，输出为 `dets` 和 `labels`。

### 2. FP32 TensorRT 推理基线

```bash
conda activate TensorRT
trtexec --onnx=python_infer/output/rfdetr-xlarge.onnx \
  --saveEngine=model_xlarge.engine --skipInference
```

`tensort_quant/build_int8_engine.py` 是可复用的 engine 构建封装；`tensorrt_convert/onnx2engine_fp.py` 是历史 FP32 转换脚本。`evaluate_coco_tensorrt.py` 负责统一预处理、TensorRT 推理、后处理和 COCOeval：

```bash
conda activate torch
python evaluate_coco_tensorrt.py --engine model_xlarge.engine \
  --output coco_predictions.json --report coco_baseline.json
```

### 3. 全计算图分析

```bash
conda activate TensorRT
trtexec --loadEngine=tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.engine \
  --iterations=200 --warmUp=200 --profilingVerbosity=detailed --dumpProfile \
  --exportProfile=tensorrt_quant_decoder/compute_graph_analysis/baseline_profile.json

conda activate torch
python tensorrt_quant_decoder/analyze_compute_distribution.py
```

该阶段输出 encoder、decoder、head 的理论 Dense FLOPs、TensorRT 融合 layer 时间、CSV 和图表，详见 [计算量报告](tensorrt_quant_decoder/compute_graph_analysis/COMPUTE_DISTRIBUTION_REPORT.md)。

### 4. Encoder 量化

`tensort_quant/quantize_int8.py`：基于 ONNX Runtime 的静态 Q/DQ PTQ，支持 MinMax、Percentile、显式节点选择和校准 cache。

```bash
conda activate torch
python tensort_quant/quantize_int8.py \
  --model python_infer/output/rfdetr-xlarge.onnx \
  --images /home/rton/val2017/val2017 \
  --output <output.onnx> --manifest <manifest.json> \
  --calibration-count 256 --quantize-ops MatMul
```

`tensort_quant/modelopt_smoothquant.py`：使用 ModelOpt 对 DINOv2 encoder Linear 的输入和权重执行 SmoothQuant，再导出 ONNX。当前推荐 encoder 产物为 `artifacts_xlarge_smoothquant_256`。

`tensort_quant/percentile_clip_encoder.py`：复用 encoder 权重 Q/DQ 图，只重统计 activation 的 p99.99 scale，避免 XLarge 直方图校准的内存开销。

### 5. Encoder 误差诊断

```bash
conda activate TensorRT
python tensort_quant/error_analysis_Conv_MatMul_compare/analyze_quant_error.py

conda activate torch
python tensort_quant/error_analysis_encoder_compare/analyze_encoder_algorithms.py
python tensort_quant/error_analysis_encoder_compare/analyze_weight_distributions.py
python tensort_quant/error_analysis_encoder_compare/analyze_activation_input_distributions.py
```

`analyze_quant_error.py` 使用 Polygraphy 的 `OnnxrtRunner`、`TrtRunner`、`Comparator` 和 `ModifyOutputs`，定位共同中间张量的误差传播。三个 encoder 分布脚本分别分析逐层算法差异、权重分布和输入激活分布。

Encoder 阶段报告入口：

| 报告 | 内容 |
|---|---|
| [FINAL_QUANTIZATION_REPORT.md](tensort_quant/FINAL_QUANTIZATION_REPORT.md) | 全量 Conv+MatMul INT8 失败原因、encoder 算法比较、速度和部署结论 |
| [STAGE_REPORT_ENCODER.md](tensort_quant/error_analysis_encoder_compare/STAGE_REPORT_ENCODER.md) | encoder 逐层误差、权重分布、输入激活误差和图表解读 |
| [INT8_QUANTIZATION_REPORT.md](tensort_quant/INT8_QUANTIZATION_REPORT.md) | 基础 PTQ、Conv-only 结果、COCO 精度和 TensorRT 延迟 |
| [PTQ_EXPERIMENT_REPORT.md](tensort_quant/PTQ_EXPERIMENT_REPORT.md) | Conv、DINO FFN、QKV/QKVO、decoder 和 SmoothQuant 分组实验 |
| [Polygraphy 诊断说明](tensort_quant/error_analysis_Conv_MatMul_compare/README.md) | 全量量化的逐层中间张量和最终输出误差定位 |

报告中引用的 encoder 图表位于 `tensort_quant/error_analysis_encoder_compare/`；全量 MinMax 和 SmoothQuant 的 Polygraphy 图表分别位于 `error_analysis_Conv_MatMul_compare/` 和 `error_analysis_Conv_MatMul_compare_smoothquant/`。

### 6. Decoder 量化

```bash
conda activate torch
python tensorrt_quant_decoder/quantize_decoder_scope.py --scope self_qkv

# 当前推荐的组合方案：encoder SQ256 + decoder Linear SmoothQuant
python tensorrt_quant_decoder/smoothquant_decoder.py \
  --scope all_linear --calibration-count 256

conda activate TensorRT
python tensort_quant/build_int8_engine.py \
  --onnx tensorrt_quant_decoder/artifacts/smoothquant_all_linear_256/rfdetr-xlarge-decoder-smoothquant-all-linear-256.onnx \
  --engine tensorrt_quant_decoder/artifacts/smoothquant_all_linear_256/rfdetr-xlarge-decoder-smoothquant-all-linear-256.engine
```

`quantize_decoder_scope.py` 用显式 decoder 节点名做 MinMax PTQ，D1～D9 用于敏感性实验；`smoothquant_decoder.py` 保留 encoder SQ256 配置，同时对选定 decoder Linear 的输入和权重启用 ModelOpt SmoothQuant。LayerNorm、残差、Softmax、deformable sampling、GridSample 和 detection head 保持原精度。

```bash
conda activate torch
python tensorrt_quant_decoder/summarize_decoder_experiments.py
```

该脚本汇总完整 COCO mAP、延迟、engine 大小并生成 decoder 敏感性图表。详细说明见 [decoder 实验报告](tensorrt_quant_decoder/DECODER_EXPERIMENT_REPORT.md) 和 [decoder 方案](tensorrt_quant_decoder/DECODER_OPTIMIZATION_PLAN.md)。

## 目录说明

| 路径 | 作用 |
|---|---|
| `python_infer/` | PyTorch 模型加载和 ONNX 导出 |
| `tensorrt_convert/` | FP32 ONNX → TensorRT 的旧版转换脚本 |
| `tensorrt_infer/` | C++ TensorRT benchmark 和端到端推理 |
| `tensort_quant/` | encoder/PTQ/SmoothQuant/Polygraphy 分析脚本和阶段报告 |
| `tensorrt_quant_decoder/` | decoder 敏感性、SmoothQuant、计算量和最终报告 |
| `evaluate_coco_tensorrt.py` | 统一 COCO TensorRT 评测入口 |
| `BASELINE.md` | FP32 TensorRT 基线记录 |
| `tensort_quant/FINAL_QUANTIZATION_REPORT.md` | encoder 和全量量化的最终诊断报告 |
| `tensorrt_quant_decoder/compute_graph_analysis/` | encoder/decoder/head 计算量与 TensorRT profile |
| `tensort_quant_decoder/artifacts/` | 本地生成的 decoder 模型、engine、预测和日志，不上传 Git |

## Git 管理原则

代码、配置、脚本和 Markdown 报告进入 Git；ONNX、`.engine`、`.pth`、COCO 逐图预测、校准缓存、TensorRT profile 原始产物和临时输出默认被 `.gitignore` 排除。重新生成这些文件时不需要修改 Git 历史。
