# Decoder 量化阶段实验报告

本报告记录在固定 encoder SmoothQuant（SQ256）基础上，对 decoder 分组进行 INT8 敏感性实验。MinMax 组使用 ONNX Runtime static PTQ/QDQ：激活采用 MinMax 校准，权重采用 per-channel INT8；SmoothQuant 组使用 ModelOpt SmoothQuant，校准图片为 256 张。

## 1. 先盘点 decoder 计算图

decoder 有 5 个 layer：`/transformer/decoder/layers.0` 到 `layers.4`。每层包含下面这些子模块：

| 子模块 | 每层包含的层/算子 | 本阶段默认处理方式 |
|---|---|---|
| Self-attention | Q/K/V projection、QKᵀ、Softmax、AV、output projection/Gemm、LayerNorm、residual Add | Q/K/V、output projection 可作为候选；QKᵀ、Softmax、AV、LayerNorm、residual 默认保护 |
| Deformable cross-attention | value projection、attention weights、sampling offsets、GridSample/采样、output projection | value/output、weights、offsets 分开测试；采样和 GridSample 默认保护 |
| FFN | `linear1`、激活、`linear2`、LayerNorm、residual Add | `linear1`、`linear2` 分开测试，再测试组合 |
| Reference point head | 两个 Linear/MatMul、Sigmoid 和坐标变换 | 默认保护 |
| Detection head | class/bbox 输出头 | 默认保护 |

## 2. 各方案实际量化范围

下面的“未量化”表示节点自身没有插入 INT8 Q/DQ；它仍可能接收来自上游量化节点的误差。

| 方案 | 实际 INT8 节点 | 明确保留 FP32/原精度的 decoder 节点 | encoder 状态 |
|---|---|---|---|
| Baseline | decoder 不新增 INT8 | self-attention、cross-attention、FFN、reference point、LayerNorm、residual、Softmax、GridSample、heads 全部保持原精度 | encoder 已是 SmoothQuant SQ256 |
| D1 Self QKV MinMax | 5 层 × self-attention Q/K/V projection，共 15 个 MatMul | self-attention QKᵀ、AV、output projection/Gemm；所有 cross-attention、FFN、reference point、LayerNorm、residual、Softmax、GridSample、heads | encoder SQ256 |
| D2 Self QKVO MinMax | D1 的 15 个 Q/K/V 加 5 个 self-attention output projection，共 20 个节点 | QKᵀ、AV、Softmax；cross-attention、FFN、reference point、LayerNorm、residual、GridSample、heads | encoder SQ256 |
| D3 FFN FC1 MinMax | 5 层 `linear1/MatMul`，共 5 个节点 | `linear2`；self-attention、cross-attention、reference point、LayerNorm、residual、Softmax、GridSample、heads | encoder SQ256 |
| D4 FFN FC2 MinMax | 5 层 `linear2/MatMul`，共 5 个节点 | `linear1`；self-attention、cross-attention、reference point、LayerNorm、residual、Softmax、GridSample、heads | encoder SQ256 |
| D5 FFN MinMax | 5 层 `linear1` + 5 层 `linear2`，共 10 个节点 | self-attention、cross-attention、reference point、LayerNorm、residual、Softmax、GridSample、heads | encoder SQ256 |
| D6 Cross Value/Output MinMax | 5 层 value projection + 5 层 output projection，共 10 个节点 | attention weights、sampling offsets、采样/GridSample；self-attention、FFN、reference point、LayerNorm、residual、heads | encoder SQ256；共享 encoder memory 可能被 activation Q/DQ 影响 |
| D7 Attention Weights MinMax | 5 层 cross-attention `attention_weights/MatMul` | value/output、sampling offsets、采样/GridSample；self-attention、FFN、reference point、LayerNorm、residual、heads | encoder SQ256 |
| D8 Sampling Offsets MinMax | 5 层 cross-attention `sampling_offsets/MatMul` | value/output、attention weights、采样/GridSample；self-attention、FFN、reference point、LayerNorm、residual、heads | encoder SQ256 |
| D9 Selective MinMax | D1+D2 的 self QKV/output、D5 FFN、D6 value/output，共 40 个显式 decoder节点 | QKᵀ、AV、Softmax、attention weights、sampling offsets、GridSample、reference point、heads | encoder SQ256 |
| Decoder all Linear SmoothQuant | ModelOpt 对 decoder Linear 的 value/output、attention weights、sampling offsets、`linear1/linear2` 做输入/权重 SmoothQuant；packed self-attention projection 不启用 | self-attention packed `in_proj_weight`、QKᵀ、AV、Softmax、LayerNorm、residual、GridSample、reference point、class/bbox heads | encoder SQ256 同时保留 |

## 3. 结果表

| 实验 | 算法与量化范围 | 图片数 | mAP@[0.50:0.95] | 相对基线 | 推理均值(ms) | 引擎(MiB) |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Encoder SmoothQuant SQ256; decoder FP32 | 5000 | 0.580251 | +0.000000 | 18.259 | 202.7 |
| D1 Self QKV | Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; self-attention Q/K/V | 5000 | 0.580645 | +0.000395 | 18.581 | 188.9 |
| D2 Self QKVO | Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; Q/K/V/output | 5000 | 0.580390 | +0.000139 | 18.352 | 185.2 |
| D3 FFN FC1 | Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; linear1 | 5000 | 0.580056 | -0.000195 | 18.301 | 188.1 |
| D4 FFN FC2 | Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; linear2 | 5000 | 0.578273 | -0.001978 | 18.105 | 187.7 |
| D5 FFN | Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; linear1/linear2 | 5000 | 0.576565 | -0.003686 | 17.746 | 172.8 |
| D6 Cross Value/Output | Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; value/output | 5000 | 0.580649 | +0.000399 | 18.573 | 195.6 |
| D7 Attention Weights | Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; attention_weights | 5000 | 0.580713 | +0.000463 | 18.485 | 202.2 |
| D8 Sampling Offsets | Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; sampling_offsets | 5000 | 0.579728 | -0.000523 | 18.499 | 201.1 |
| D9 Selective | Encoder SmoothQuant SQ256 + decoder static PTQ QDQ MinMax; QKV/QKVO/FFN/value/output | 5000 | 0.576634 | -0.003616 | 17.988 | 148.0 |
| SQ Decoder all Linear | ModelOpt SmoothQuant; encoder + decoder Linear; 256 images | 5000 | 0.580280 | +0.000029 | 18.710 | 163.8 |

完整 COCO 结果优先使用 5000 张验证集；只有尚未完成完整评估的实验才保留 100 张预筛选结果。`delta_map` 相对于 baseline_encoder_sq256 计算。

## 4. 如何读取敏感性

D1、D2、D3、D6、D7 在完整数据上与 baseline 接近，说明对应 decoder 子图的 INT8 误差仍能被后续网络吸收。D4 有可见但有限的下降，D5 和联合 D9 的下降明显，说明逐层单独可接受不代表联合量化一定可接受。最终应以完整 COCO 为准，100 张只用于快速筛选。

D6 的额外诊断很关键：cross-attention 的 `value_proj` 输入来自 encoder memory，同时还被 `enc_out_bbox_embed` 等 encoder 输出分支共享。静态 QDQ 在该共享张量上插入 activation Quantize/Dequantize 后，虽然目标节点属于 decoder，encoder-to-decoder 接口及旁路 head 也会受到影响。因此 D6 的逐层表中可能在 `enc_out_bbox_embed` 处先出现误差；这不是权重被误改，而是共享激活边界被量化。若要把 D6 变成严格的“仅 decoder”实验，应先复制该共享输入或采用只量化权重/只在 decoder 分支量化 activation 的图改写。

## 5. Polygraphy 逐层误差分析

每个 D1、D3、D6、D9 目录下的 `error_analysis/` 由 Polygraphy `OnnxrtRunner` 在相同固定输入上导出共同 MatMul/Gemm 输出，并生成 `layerwise_errors.csv`、`summary.json`、`plots/layerwise_error_propagation.png`、`plots/top15_relative_l1.png`、`plots/error_by_op_type.png` 和 Q/DQ scale 审计。横轴 `order` 是 ONNX 拓扑顺序，纵轴 `relative_l1` 是平均绝对误差除以参考张量平均绝对值；它用于定位误差第一次明显上升的位置，不等于最终 mAP 损失。

Polygraphy 的 TensorRT engine 对比还会输出 `polygraphy_error_metrics/`。它比较 baseline engine 和候选 engine 的 `dets`、`labels`，用于确认最终输出是否敏感；中间层定位仍以 ONNX Runtime 共同张量为主。

## 6. SmoothQuant 说明

`smoothquant_decoder.py` 保留 encoder SQ256 配置，同时只启用指定 decoder Linear 模块的输入和权重量化器。self-attention 的 packed `in_proj_weight`、LayerNorm、残差、softmax、deformable sampling 和检测头保持 FP32。ModelOpt 导出和 TensorRT 构建均已通过；完整 COCO 评估可在该目录继续运行。

## 7. 产物目录

- `artifacts/baseline_encoder_sq256/`：固定 encoder SmoothQuant、decoder FP32 基线。
- `artifacts/d0*_minmax/`：D1-D9 的 decoder 分组 MinMax QDQ ONNX、TensorRT engine、校准 manifest、100/5000 图片评估和逐层误差分析。
- `artifacts/smoothquant_all_linear_256/`：ModelOpt SmoothQuant decoder Linear 导出模型和 engine。
- `calibration_images_256.txt`：256 张校准图片清单。
- `decoder_sensitivity_summary.csv`：可被表格软件直接排序的统一结果。
- `plots/`：mAP、延迟和精度/延迟散点图。
