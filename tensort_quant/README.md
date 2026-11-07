# Encoder PTQ 与误差诊断

本目录负责 encoder 阶段的 PTQ、SmoothQuant 和 Polygraphy 误差分析。整个工程的总流程见仓库根目录 [README.md](../README.md)；本目录的综合结论见 [FINAL_QUANTIZATION_REPORT.md](FINAL_QUANTIZATION_REPORT.md)。

## 脚本说明

| 文件 | 作用 |
|---|---|
| `quantize_int8.py` | ONNX Runtime 静态 Q/DQ PTQ；支持 MinMax、Percentile、显式节点选择和 calibration cache |
| `modelopt_smoothquant.py` | ModelOpt encoder SmoothQuant 导出；默认只处理 DINOv2 encoder Linear |
| `percentile_clip_encoder.py` | 低内存 activation p99.99 scale 重统计，复用已有权重 Q/DQ 图 |
| `build_int8_engine.py` | Q/DQ ONNX → TensorRT engine 的 `trtexec` 封装 |
| `run_pipeline.sh` | Conv-only 的完整 PTQ、engine、benchmark 和 COCO 评测流程 |
| `error_analysis_Conv_MatMul_compare/analyze_quant_error.py` | Polygraphy/ONNX Runtime 逐层误差、Q/DQ scale 和最终输出比较 |
| `error_analysis_encoder_compare/analyze_encoder_algorithms.py` | MinMax、Percentile、SmoothQuant 的逐层误差对比 |
| `error_analysis_encoder_compare/analyze_weight_distributions.py` | 权重量化前后分布、误差和饱和率 |
| `error_analysis_encoder_compare/analyze_activation_input_distributions.py` | MatMul 输入激活分布、上游累计误差、当前 Q/DQ 新增误差 |

## 常用命令

量化脚本需要 `torch` 环境，因为它依赖 ONNX、OpenCV 和 ModelOpt；TensorRT engine 构建和 Polygraphy 诊断使用 `TensorRT` 环境。

```bash
cd /home/rton/rfdetr

conda activate torch
python tensort_quant/quantize_int8.py \
  --model python_infer/output/rfdetr-xlarge.onnx \
  --images /home/rton/val2017/val2017 \
  --output /tmp/rfdetr-encoder-int8.onnx \
  --manifest /tmp/rfdetr-encoder-manifest.json \
  --calibration-count 256 --quantize-ops MatMul

conda activate TensorRT
python tensort_quant/build_int8_engine.py \
  --onnx /tmp/rfdetr-encoder-int8.onnx \
  --engine /tmp/rfdetr-encoder-int8.engine

python tensort_quant/error_analysis_Conv_MatMul_compare/analyze_quant_error.py
```

## 证据目录

- `FINAL_QUANTIZATION_REPORT.md`：全量 INT8 失败原因、encoder 算法比较和部署建议。
- `INT8_QUANTIZATION_REPORT.md`：基础 XLarge PTQ、Conv-only 结果和速度对照。
- `PTQ_EXPERIMENT_REPORT.md`：早期 Conv、decoder、DINO FFN/QKV/QKVO 分组实验。
- `CALIBRATION_DATA.md`：校准数据采样和预处理说明。
- `error_analysis_Conv_MatMul_compare/`：全量量化失败的 Polygraphy 图表和说明。
- `error_analysis_Conv_MatMul_compare_smoothquant/`：SmoothQuant 对照图表。
- `error_analysis_encoder_compare/`：encoder 权重和输入激活分布图表及分析脚本。

`artifacts_xlarge*/`、ONNX、TensorRT engine、校准缓存、COCO 逐图预测和原始 CSV/JSON 由 `.gitignore` 排除，只在本地实验时生成。最终数值已整理进 Markdown 报告和 decoder 汇总表。
