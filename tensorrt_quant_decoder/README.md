# RF-DETR XLarge Decoder 优化工程

当前阶段的方案入口是 [`DECODER_OPTIMIZATION_PLAN.md`](DECODER_OPTIMIZATION_PLAN.md)。

本目录以已经完成的 SmoothQuant encoder 256 张校准结果为固定前提，逐步分析和量化 decoder。所有新候选都应从下面的 encoder ONNX 开始：

```text
/home/rton/rfdetr/tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx
```

已有的 decoder MinMax 64 张实验位于：

```text
/home/rton/rfdetr/tensort_quant/artifacts_xlarge_decoder/
```

该结果使用原始 FP32 encoder，主要用于历史敏感性参考，不能直接作为本目录的最终基线。

本目录计划放置：

- decoder 分组量化脚本和 SmoothQuant 导出脚本；
- 固定 256 张校准图片清单；
- Polygraphy 逐层误差诊断脚本；
- 每个候选的 ONNX、TensorRT engine、COCO 评测和 benchmark；
- decoder 误差图、分布 CSV/JSON 和阶段性报告。

当前已执行的入口：

```bash
# D1-D9：在固定 SQ256 encoder ONNX 上做 decoder 分组 MinMax PTQ
/home/rton/miniconda3/envs/torch/bin/python quantize_decoder_scope.py --scope self_qkv

# ModelOpt SmoothQuant：encoder SQ256 + 选定 decoder Linear
/home/rton/miniconda3/envs/torch/bin/python smoothquant_decoder.py --scope all_linear --calibration-count 256

# TensorRT 构建（TensorRT 环境）
/home/rton/miniconda3/envs/TensorRT/bin/python ../tensort_quant/build_int8_engine.py --onnx <model.onnx> --engine <model.engine>

# 逐层 Polygraphy/ONNX Runtime + TensorRT 输出诊断
/home/rton/miniconda3/envs/TensorRT/bin/python ../tensort_quant/error_analysis_Conv_MatMul_compare/analyze_quant_error.py \
  --fp32-onnx ../tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx \
  --int8-onnx <candidate.onnx> --fp32-engine artifacts/baseline_encoder_sq256/rfdetr-xlarge-sq256-decoder-fp32.engine \
  --int8-engine <candidate.engine> --inputs ../tensort_quant/polygraphy_analysis/.inputs.json \
  --output-dir <candidate>/error_analysis --op-types MatMul,Gemm

# 统一生成 CSV、PNG 和 Markdown 报告
/home/rton/miniconda3/envs/torch/bin/python summarize_decoder_experiments.py

# 全计算图 encoder/decoder/head 计算量与 TensorRT profile
/home/rton/miniconda3/envs/TensorRT/bin/trtexec \
  --loadEngine=../tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.engine \
  --iterations=200 --warmUp=200 --profilingVerbosity=detailed --dumpProfile \
  --exportProfile=compute_graph_analysis/baseline_profile.json
/home/rton/miniconda3/envs/torch/bin/python analyze_compute_distribution.py

结果写入 `compute_graph_analysis/COMPUTE_DISTRIBUTION_REPORT.md`，其中同时给出
ONNX 的 Conv/MatMul/Gemm 理论 FLOPs 和 TensorRT 融合 layer 的实际 profile 时间。
```

其中 `analyze_quant_error.py` 的 `OnnxrtRunner`、`TrtRunner`、`Comparator` 和
`save_error_metrics_plot` 均来自 Polygraphy；`error_analysis/plots/` 保存逐层误差、
Top-15 张量、算子类型箱线图和 Q/DQ scale 直方图。

执行环境：

```bash
conda activate TensorRT
cd /home/rton/rfdetr
```
