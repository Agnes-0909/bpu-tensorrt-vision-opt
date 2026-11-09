# Stage 2：高级计算图与端侧运行时优化

这个目录承接第一阶段的结果。第一阶段已经完成：encoder 使用 ModelOpt SmoothQuant、256 张校准图片，decoder 选择了当前精度最稳的组合；当前全图仍然需要继续降低端到端耗时。这里先完成热点定位、候选方案设计和可复现实验入口，再逐项生成新的 TensorRT engine。

## 目录内容

| 文件 | 作用 | 运行方式 |
|---|---|---|
| [`STAGE2_ADVANCED_OPTIMIZATION_PLAN.md`](STAGE2_ADVANCED_OPTIMIZATION_PLAN.md) | 阶段二的结论、方案排序、风险和验收标准 | 直接阅读 |
| [`STAGE2_EXPERIMENT_REPORT.md`](STAGE2_EXPERIMENT_REPORT.md) | 第一轮 attention 改写、Polygraphy、COCO、TensorRT profile 和持久 runtime 实验结果 | 直接阅读 |
| [`profile_hotspots.py`](profile_hotspots.py) | 从 TensorRT profile CSV 聚合 Softmax、QKᵀ、AV、归一化和采样热点，并生成图表 | `python profile_hotspots.py --profile-csv ../tensorrt_quant_decoder/compute_graph_analysis/tensorrt_profile_layers.csv --out-dir analysis_outputs` |
| [`inspect_attention_patterns.py`](inspect_attention_patterns.py) | 只读检查 ONNX 注意力节点，确认是否满足 TensorRT attention 融合改写条件 | `conda run -n TensorRT python inspect_attention_patterns.py <model.onnx>` |
| [`experiments/fused_attention_plugin/`](experiments/fused_attention_plugin/) | 自定义 attention plugin 和 PyTorch SDPA/FlashAttention 原型：图替换、CUDA kernel、TensorRT 动态注册和中间张量校验 | 先阅读目录 README，再按其中的 `nvcc`/`trtexec` 命令运行 |
| [`summarize_stage2_experiment.py`](summarize_stage2_experiment.py) | 汇总 `trtexec`、C++ benchmark 和 detailed profile，生成 CSV、延迟图和时间分解图 | `python summarize_stage2_experiment.py` |
| [`figures/`](figures/) | 报告中直接引用的三张轻量结果图；由汇总脚本生成后复制保存 | 直接查看 |
| `analysis_outputs/` | 本地生成的热点 CSV、JSON、PNG；不作为模型或实验缓存上传 | 由脚本生成 |

后续实验脚本和 engine 应放在本目录下的独立子目录，命名带上精度、校准集和范围，例如 `experiments/attention_fp16_fused/`、`experiments/attention_int8_score_fp16_softmax/`。ONNX、engine、校准缓存、COCO 预测结果和 profile 原始 JSON 均属于本地产物。

## 当前基线

- 参考 engine：`tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.engine`
- 参考 ONNX：同目录下同名 `.onnx`
- 输入：`1×3×700×700`，batch=1
- GPU/软件：RTX 5060 Ti、TensorRT 11.4
- profile 原始数据：`tensorrt_quant_decoder/compute_graph_analysis/tensorrt_profile_layers.csv`

本目录不直接修改第一阶段 engine。每个候选要同时保存构图参数、精度范围、profile、端到端 benchmark 和 COCO 结果，确保收益可以追溯。

当前实验结论见 [`STAGE2_EXPERIMENT_REPORT.md`](STAGE2_EXPERIMENT_REPORT.md)：ONNX `Attention` 能被 TensorRT 解析，但没有在当前 GPU/shape 上形成真正 fused kernel；C++ 持久 stream/device buffer 可直接采用。FlashAttention 原型把纯 GPU latency 降到约 8.03 ms，完整 COCO mAP 为 0.5803911，但当前实现仍依赖 libtorch 运行时。
