# RF-DETR XLarge 全计算图计算量分布

## 1. 统计对象

模型：`/home/rton/rfdetr/tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx`；输入固定为 `1×3×700×700`。ONNX 统计以 Conv/MatMul/Gemm 为主，乘加按 2 FLOPs 计；TensorRT 统计来自 `trtexec --profilingVerbosity=detailed --dumpProfile`，并把融合 kernel 按计算图连续区域归属到 encoder、decoder、head。

## 2. 结果

| 阶段 | ONNX dense FLOPs | FLOPs 占比 | TensorRT profile 时间 | profile 时间占比 |
|---|---:|---:|---:|---:|
| encoder | 280.06 GFLOPs | 95.7% | 14.936 ms | 85.1% |
| decoder | 11.74 GFLOPs | 4.0% | 2.260 ms | 12.9% |
| head | 0.77 GFLOPs | 0.3% | 0.351 ms | 2.0% |

## 3. encoder/decoder 内部构成

| 阶段 | 子模块 | dense FLOPs | 阶段内占比 |
|---|---|---:|---:|
| encoder | FFN fc1 | 69.42 GFLOPs | 24.8% |
| encoder | FFN fc2 | 69.42 GFLOPs | 24.8% |
| encoder | attention QKV projection | 52.06 GFLOPs | 18.6% |
| encoder | attention AV | 27.70 GFLOPs | 9.9% |
| encoder | attention QK^T | 27.70 GFLOPs | 9.9% |
| encoder | attention output projection | 17.35 GFLOPs | 6.2% |
| encoder | feature projector | 14.13 GFLOPs | 5.0% |
| encoder | other | 2.26 GFLOPs | 0.8% |
| decoder | FFN | 6.29 GFLOPs | 53.6% |
| decoder | self-attention | 4.07 GFLOPs | 34.7% |
| decoder | deformable cross-attention | 1.38 GFLOPs | 11.7% |
| decoder | reference-point head | 0.00 GFLOPs | 0.0% |
| head | other | 0.43 GFLOPs | 55.5% |
| head | decoder bbox head | 0.32 GFLOPs | 40.8% |
| head | decoder class head | 0.03 GFLOPs | 3.6% |
| head | encoder output projection | 0.00 GFLOPs | 0.2% |

## 4. 如何解读

- FLOPs 表示理论计算量，回答“需要做多少乘加”；TensorRT profile 表示实际 kernel 时间，回答“GPU 在这个版本上花了多少时间”。两者不会严格成正比，因为 TensorRT 会融合算子、使用不同精度 kernel，并受到访存、同步和 plugin 影响。
- encoder 包含 DINOv2 backbone 和多尺度 feature projector；这里的 attention 是规则的大矩阵计算，重复 block 多，因此既有最大的 dense FLOPs，也有最大的实际耗时占比。
- encoder 内部 FLOPs 最大的是 FFN fc1/fc2，其次是 QKᵀ、AV 和 QKV projection；这解释了为什么 encoder Linear/attention projection 的 INT8 化能够影响整体速度。
- decoder 只有 5 层、query 数量较小，并且 deformable cross-attention 中包含采样、GridSample、Gather 和残差归一化；其中很多操作不是大规模规则 GEMM，所以 decoder FLOPs 和可被 INT8 加速的 FLOPs 都有限。
- head 包含 encoder 输出筛选、TopK、reference/bbox/class 分支以及最终检测头。它的 dense FLOPs 很小，但包含后处理相关的图操作，不能只用 FLOPs 判断其时间。

## 5. 与量化收益的关系

当前 encoder SmoothQuant 的速度收益主要来自覆盖了计算图中占比最大的规则 dense encoder 计算；decoder 量化只覆盖较小的 GEMM 子集，Softmax、deformable sampling、GridSample、LayerNorm 和残差仍然保留，因此完整图推理收益自然有限。

## 6. 产物

- `compute_distribution_summary.json`：完整机器可读统计。
- `onnx_compute_nodes.csv`：逐 ONNX 节点的 stage、算子、shape 和 FLOPs。
- `tensorrt_profile_layers.csv`：逐 TensorRT 融合 layer 的平均时间和占比。
- `plots/onnx_flops_by_stage.png`：理论 dense FLOPs 分布。
- `plots/tensorrt_time_by_stage.png`、`tensorrt_time_share.png`：实际 profile 时间分布。
