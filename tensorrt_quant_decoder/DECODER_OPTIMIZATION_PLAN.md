# RF-DETR XLarge Decoder 量化与推理优化方案

## 1. 当前状态和目标

Encoder 阶段已经确定采用：

- 算法：ModelOpt SmoothQuant；
- 校准图片：256 张；
- 产物：`/home/rton/rfdetr/tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx`；
- 完整 COCO mAP@[0.50:0.95]：0.5802505；
- 纯 TensorRT 图推理：约 16.24 ms；
- 应用级单图 inference：约 18.83 ms。

下一阶段只优化 Decoder，目标是：

1. 保持已经完成的 SmoothQuant encoder 结果不变；
2. 找到 decoder 中可以安全 INT8 化的算子；
3. 识别 self-attention、deformable cross-attention、FFN 和预测头的敏感性差异；
4. 在精度几乎不下降的前提下继续降低 TensorRT 图推理时间和 engine 大小；
5. 形成可以复现的 decoder 量化模型、逐层误差分析、COCO 指标和速度对照。

本阶段工程目录为：

```text
/home/rton/rfdetr/tensorrt_quant_decoder/
```

当前该目录为空；后续脚本、实验清单、图表和最终报告都放在这里。

## 2. 先区分两个已有结果

已有目录中的：

```text
tensort_quant/artifacts_xlarge_decoder/
```

是旧的 **decoder-only MinMax 64 张校准**实验。它的源图是原始 FP32 ONNX，并且通过排除 `/backbone/` 保留了 FP32 encoder。因此：

- mAP@[0.50:0.95] = 0.5767576；
- 相对 FP32 baseline 下降约 0.46 个百分点；
- COCO evaluator inference 平均约 25.52 ms；
- 该结果量化了 decoder 以外的部分 transformer 输出/参考点 MatMul；
- 它不能直接回答“SmoothQuant encoder + decoder INT8”最终会是多少。

新阶段的唯一公平基线应当是：

```text
SmoothQuant encoder 256 + FP32 decoder
= artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx
```

后续所有 decoder 候选都从这张图开始，使用相同的输入、相同的后处理、相同的 256 张校准图片和相同的 TensorRT 构建参数。

## 3. Decoder 计算图盘点

原始 ONNX 中 decoder 有 5 个 layer：

```text
/transformer/decoder/layers.0
...
/transformer/decoder/layers.4
```

每层包含：

| 子模块 | 每层算子 | 5 层总数 | 初始判断 |
|---|---:|---:|---|
| self-attention | Q/K/V、QKᵀ、AV 等 5 个 MatMul + 1 个 Gemm | 30 个 MatMul/Gemm | Q/K/V 和 output projection 候选；QKᵀ、AV 先保护 |
| deformable cross-attention | value projection、attention weights、sampling offsets、output projection 共 4 个 MatMul | 20 个 MatMul | value/output projection 候选；weights/offsets 先保护 |
| FFN | linear1、linear2 共 2 个 MatMul | 10 个 MatMul | 先做独立敏感性实验 |
| reference point head | 2 个 MatMul | 2 个 MatMul | 几何参考点，默认保护 |
| normalization/residual/Softmax/GridSample | Add、LayerNorm、Softmax、GridSample 等 | 多个 | 默认保持 FP16/FP32 |
| 分类/回归输出头 | class/bbox MatMul/Gemm | 多个 | 默认保持 FP32，避免输出阈值和坐标分布变化 |

已有 decoder MinMax 产物的 Q/DQ 审计显示：每个 decoder layer 的 11 个 MatMul 都被量化，但 self-attention 的 Gemm 没有被量化；同时 reference point head 和 `transformer/enc_output.0` 也被量化。也就是说，旧实验的“decoder-only”范围比名称看起来更宽，并且把 attention score、deformable attention geometry 和 reference point 一起放进去了。这是新阶段必须拆分的原因。

## 4. 总体策略：先定位，再用 SmoothQuant 收敛

不要直接把整个 decoder 一次性 SmoothQuant。建议使用两步法：

### 第一步：用窄范围 MinMax 做敏感性定位

MinMax 只作为诊断基线，不作为最终算法。它的优点是：

- 可以用现有 `quantize_int8.py` 快速生成 QDQ 图；
- 每次只改变一组 decoder 节点；
- 便于比较“某一类节点是否本身敏感”；
- 可以把误差归因到 self-attention、cross-attention、FFN 或预测头。

### 第二步：只对通过筛选的 Linear/Projection 使用 SmoothQuant

SmoothQuant 只应用到第一步证明安全的线性投影：

- decoder self-attention 的 Q/K/V projection；
- self-attention output projection；
- cross-attention value/output projection；
- decoder FFN linear1/linear2 中经过验证的部分。

以下算子先不做 SmoothQuant：

- self-attention 的 QKᵀ score；
- self-attention 的 AV；
- cross-attention 的 attention weights；
- cross-attention 的 sampling offsets；
- Softmax；
- GridSample；
- LayerNorm、residual Add；
- reference point head；
- class/bbox 输出头。

原因是这些算子分别控制注意力排序、采样位置、归一化和最终检测坐标。它们的误差不是普通矩阵乘法误差，应该在逐层诊断证明安全后再考虑。

## 5. 推荐实验矩阵

所有实验固定：

- 源图：SmoothQuant encoder 256 ONNX；
- 校准图片：SmoothQuant 256 的同一份 manifest；
- 输入预处理：与现有 COCO evaluator 完全一致；
- 校准算法：先 MinMax 定位，再对候选组使用 SmoothQuant；
- 评测：5000 张 COCO val2017；
- 中间层诊断：固定 Polygraphy 输入，同时比较 FP32 decoder reference 和候选图；
- 每个候选都记录 mAP、mAP50、small/medium/large、engine 大小、应用延迟和纯 TensorRT 图延迟。

### 5.1 第一轮：单组敏感性实验

| 编号 | 量化范围 | 保护范围 | 目的 |
|---|---|---|---|
| D0 | 不新增 decoder INT8 | 整个 decoder FP32 | 新阶段基线 |
| D1 | 5 层 self-attention Q/K/V projection | QKᵀ、AV、Gemm/output、cross-attn、FFN、heads FP32 | 判断 QKV 是否安全 |
| D2 | D1 + self-attention output projection/Gemm | QKᵀ、AV 及其它模块 FP32 | 判断 self-attention 输出投影 |
| D3 | 5 层 FFN linear1 | 其它 decoder 模块 FP32 | 判断 FFN 第一层 |
| D4 | 5 层 FFN linear2 | 其它 decoder 模块 FP32 | 判断 FFN 第二层 |
| D5 | D3 + D4 | 其它 decoder 模块 FP32 | 判断 decoder FFN 组合误差 |
| D6 | cross-attention value projection + output projection | attention weights、sampling offsets、GridSample FP32 | 判断安全的 value/output 路径 |
| D7 | cross-attention attention_weights | sampling offsets、GridSample FP32 | 单独验证 attention weight 敏感性 |
| D8 | cross-attention sampling_offsets | attention weights、GridSample FP32 | 单独验证采样坐标敏感性 |
| D9 | D2 + D5 + D6 | score、weights、offsets、GridSample、heads FP32 | 推荐的选择性 decoder 候选 |
| D10 | 所有 decoder Linear/MatMul | score、Softmax、offset、GridSample、reference point、heads FP32 | 观察只量化线性层的上限 |

第一轮只使用 MinMax。每个实验都应生成一张独立 ONNX 和 engine，不能通过修改同一个文件覆盖结果。

### 5.2 第二轮：SmoothQuant 替换

对第一轮中满足精度门槛的组重新导出 SmoothQuant：

- SQ-D1：self-attention Q/K/V；
- SQ-D2：self-attention Q/K/V + output；
- SQ-D5：decoder FFN；
- SQ-D6：cross-attention value/output；
- SQ-D9：选择性 decoder Linear 全组合；
- SQ-D10：所有通过筛选的 decoder Linear。

第二轮的目的不是重新发现敏感层，而是检查 SmoothQuant 是否能把 MinMax 候选的 activation outlier 降低，从而提升精度或允许更大量化范围。

## 6. 诊断脚本设计

在 `tensorrt_quant_decoder/` 中新增以下文件：

```text
tensorrt_quant_decoder/
├── README.md
├── DECODER_OPTIMIZATION_PLAN.md
├── quantize_decoder_scope.py
├── smoothquant_decoder.py
├── build_decoder_engine.py
├── analyze_decoder_error.py
├── evaluate_decoder_candidates.sh
├── calibration_images_256.txt
├── artifacts/
│   ├── baseline_encoder_sq256/
│   ├── d01_self_qkv_minmax/
│   ├── d02_self_qkvo_minmax/
│   ├── d05_ffn_minmax/
│   ├── d06_cross_value_output_minmax/
│   ├── d09_selective_minmax/
│   ├── sq_d02_self_qkvo/
│   ├── sq_d05_ffn/
│   ├── sq_d06_cross_value_output/
│   └── sq_d09_selective/
└── reports/
    ├── decoder_scope_summary.csv
    ├── decoder_candidate_comparison.json
    ├── decoder_error_report.md
    └── plots/
```

### 6.1 `quantize_decoder_scope.py`

功能：

- 输入固定为 encoder SmoothQuant 256 ONNX；
- 支持 `--scope self_qkv`、`self_qkvo`、`ffn`、`cross_value_output`、`selective`；
- 支持 `MatMul,Gemm`；
- 显式 include pattern，避免误量化 `enc_output`、reference point 和 detection head；
- 校验 encoder 已有 Q/DQ 数量不变；
- 输出 manifest，记录量化节点、保护节点、校准方法和 256 张图片。

关键原则是不要只用 `--exclude-pattern /backbone/`。新脚本必须采用显式 include pattern，否则 decoder 之外的 `transformer/enc_output.0` 和 reference point/head 仍可能被带入量化。

### 6.2 `smoothquant_decoder.py`

功能：

- 复用 ModelOpt SmoothQuant；
- 先禁用所有 quantizer；
- 只启用选定 decoder Linear 的 input/weight quantizer；
- decoder 的 attention score、Softmax、GridSample、LayerNorm、residual 和 heads 默认禁用；
- 校准循环固定使用 256 张图；
- 导出后检查 ONNX 中 encoder Q/DQ、decoder Q/DQ 和输出节点是否符合 manifest。

如果 ModelOpt 无法稳定处理 deformable cross-attention 的模块边界，先只对 self-attention 和 FFN 的标准 Linear 做 SmoothQuant，cross-attention 保留 MinMax 或 FP16。

### 6.3 `analyze_decoder_error.py`

诊断对象必须是：

```text
FP32 reference:
  原始 FP32 ONNX 或 encoder SQ256 + FP32 decoder reference

Candidate:
  encoder SQ256 + decoder candidate
```

不能把 decoder candidate 直接和原始全 FP32 图比较后就下结论，因为这样会把 encoder SmoothQuant 的误差混进 decoder 误差。

建议使用 Polygraphy 标记以下中间输出：

- 每层 self-attention 的 Q/K/V projection；
- QKᵀ score；
- Softmax 后 attention weight；
- AV 输出；
- self-attention output projection；
- cross-attention value、attention weight、sampling offset、GridSample 后输出；
- linear1、activation 后、linear2；
- 每个 decoder layer residual 后输出；
- reference point；
- class/bbox head 前后输出；
- 最终 `dets`、`labels`。

输出：

- relative L1；
- RMSE；
- cosine similarity；
- p99/p99.9/p99.99；
- activation scale；
- saturation ratio；
- 误差首次出现的 decoder layer；
- 上游输入误差、当前 Q/DQ 误差、有效输入误差。

图表至少包括：

1. decoder 拓扑顺序误差传播图；
2. 按 layer 的误差 heatmap；
3. 按子模块的箱线图；
4. self-attention score/Softmax 分布；
5. cross-attention attention weights 分布；
6. sampling offsets 的绝对误差和尾部；
7. FFN linear1/linear2 激活直方图；
8. decoder residual 输出误差；
9. Q/DQ scale 和 saturation；
10. 最终 dets/labels 的误差图。

## 7. 评测和验收标准

### 精度门槛

相对于新基线“SmoothQuant encoder 256 + FP32 decoder”：

- **推荐候选**：mAP 下降不超过 0.1 个百分点；
- **可接受候选**：mAP 下降不超过 0.2 个百分点；
- **诊断候选**：下降超过 0.2 个百分点，只用于定位敏感模块；
- 任意候选若 mAP 下降超过 1 个百分点，停止继续扩大量化范围，回到逐层保护实验。

同时检查：

- mAP50 不能出现异常崩溃；
- small/medium/large 不能只有某一类尺寸大幅掉点；
- `dets` 和 `labels` 的 relative L1 不能只看总均值；
- decoder 中间层 cosine 不能在 Softmax 前后突然跌到 0.9 以下；
- sampling offsets 的 p99.99 不应出现数量级放大；
- reference point 和 bbox 输出的坐标范围不能异常扩张。

### 速度门槛

每个候选都测两类时间：

1. 纯 TensorRT 图推理：trtexec，关闭 H2D/D2H，开启 CUDA Graph；
2. 应用级推理：和现有 C++ benchmark 相同的 H2D、enqueue、D2H、buffer 管理。

只有当精度通过后，才比较速度。decoder 量化的理论收益可能小于 encoder，因为：

- decoder 只有 5 层；
- query 数量较小；
- deformable attention 中 GridSample、reshape、memory access 不是 INT8 GEMM；
- FP32/FP16 和 INT8 混合会引入 Q/DQ 转换。

因此不要以“所有 decoder MatMul 都 INT8”作为目标，而以“单位精度损失带来的有效延迟收益”作为目标。

## 8. 推荐执行顺序

### 阶段 A：建立公平基线

1. 从 SmoothQuant encoder 256 ONNX 构建 FP32 decoder baseline engine；
2. 用完整 COCO 评测确认 mAP 和延迟；
3. 将 256 张 manifest 复制为 `calibration_images_256.txt`；
4. 用 Polygraphy 保存固定 decoder 输入和中间 reference。

### 阶段 B：单模块 MinMax 敏感性

依次运行 D1、D2、D3、D4、D5、D6、D7、D8。每次只新增一类 decoder 节点，记录 mAP 和中间误差。

### 阶段 C：选择性组合

根据阶段 B 的结果构建 D9/D10。优先组合：

```text
self QKV
→ self QKVO
→ + decoder FFN
→ + cross value/output
```

任何一步超过精度门槛，都回退最后新增的模块，并对该模块按 layer 0～4 做保护实验。

### 阶段 D：SmoothQuant 替换

只对通过 MinMax 筛选的标准 Linear 使用 SmoothQuant。固定 256 张校准图，生成 SQ-D1、SQ-D2、SQ-D5、SQ-D6、SQ-D9。

### 阶段 E：最终 TensorRT 优化

在精度合格的最终候选上：

- 比较 FP16、INT8、混合精度的 tactic；
- 确认 Q/DQ 没有覆盖 Softmax、LayerNorm、GridSample；
- 启用 CUDA Graph 和固定 shape；
- 检查 workspace、timing cache 和 engine 构建日志；
- 重复 3～5 次 benchmark，报告均值和 P95；
- 只保留最终候选、失败敏感性证据和必要日志。

## 9. 当前推荐的预期结果

根据 encoder 阶段经验，推荐优先验证：

1. **SQ-D2：self-attention Q/K/V + output projection**；
2. **SQ-D5：decoder FFN linear1/linear2**；
3. **SQ-D6：cross-attention value/output projection**；
4. **SQ-D9：上述三组的选择性组合**。

暂不建议第一轮量化：

- self-attention QKᵀ；
- self-attention AV；
- cross-attention attention weights；
- sampling offsets；
- reference point head；
- detection heads。

这些位置即使权重直方图看起来正常，也可能通过 Softmax 排序或 GridSample 坐标把很小的数值误差放大成检测框变化。

## 10. 完成定义

decoder 阶段完成需要同时满足：

- 有一份固定 256 张校准图片 manifest；
- 有 D0 和至少 D1～D8 的单模块结果；
- 有 Polygraphy 逐层误差 CSV、JSON 和图表；
- 有至少一个 SmoothQuant decoder 候选；
- 有完整 COCO 5000 张评测；
- 有纯图和应用级速度；
- 报告能解释误差从哪一层开始、哪类输入分布导致、哪些模块保留 FP32/FP16；
- 最终 engine 与 ONNX、manifest、评测和 benchmark 一一对应。

在完成 D1～D8 之前，不建议直接构建“全 decoder INT8”作为最终部署方案。
