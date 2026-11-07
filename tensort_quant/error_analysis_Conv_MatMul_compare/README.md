# Conv + MatMul INT8 误差定位

这个目录用于分析 `artifacts_xlarge/rfdetr-xlarge-int8.onnx` 的全量 Conv+MatMul INT8 量化为什么会失败。脚本直接使用 Polygraphy 的 `ModifyOutputs`、`OnnxrtRunner`、`Comparator`、`StreamingDataLoader` 和 `TrtRunner`，在同一份输入上标记两张 ONNX 图中共同的 Conv/MatMul 输出，逐层计算绝对误差、RMSE、相对 L1 误差和 cosine similarity；最后再比较已经构建好的两个 TensorRT engine 的 `dets`、`labels` 输出。

当前 `artifacts_xlarge` 的结果属于：**静态 PTQ、显式 Q/DQ、ONNX Runtime MinMax 激活校准、对称 INT8 激活、权重 per-channel INT8**。`summary.json` 的 `quantization_algorithm` 字段会记录该标注；下面的 Entropy 结果使用完全相同的量化范围和输入，只替换激活校准算法。

在 TensorRT 环境中运行：

```bash
cd /home/rton/rfdetr
conda activate TensorRT
python tensort_quant/error_analysis_Conv_MatMul_compare/analyze_quant_error.py
```

默认输入是已有的 `tensort_quant/polygraphy_analysis/.inputs.json`，因此 FP32 和 INT8 使用完全相同的输入。完整分析会产生较大的中间数组，但不会把数组写入磁盘；只写入指标：

- `summary.json`：首个明显误差位置、误差最大的 tensor、按算子类型汇总、TensorRT 最终输出对比；
- `layerwise_errors.csv`：按原始图拓扑顺序排列的逐 tensor 误差，可以直接按 `order`、`relative_l1` 排序；
- `qdq_scales.csv` / `qdq_scales.json`：INT8 ONNX 中 QuantizeLinear/DequantizeLinear 的 scale、axis 和 per-channel 统计。
- `plots/layerwise_error_propagation.png`：误差随拓扑顺序传播，以及 cosine similarity 的变化；
- `plots/top15_relative_l1.png`、`plots/error_by_op_type.png`、`plots/qdq_scale_distribution.png`：Top-K、算子类型分布和 Q/DQ scale 分布；
- `polygraphy_error_metrics/error_metrics_dets.png`、`error_metrics_labels.png`：Polygraphy `Comparator` 自带的最终输出误差图。

## 图表逐张读法

`plots/layerwise_error_propagation.png` 有上下两个面板。横轴是 ONNX 拓扑顺序，不是层号；上方纵轴是 relative L1 error，使用 symlog 坐标，虚线是 0.5；下方纵轴是 cosine similarity，虚线 1 是理想值。颜色表示 Conv、MatMul 和最终 GraphOutput。先找上方第一个越过 0.5 的点，再看下方同一位置的 cosine 是否也明显下降；连续高点表示误差沿图传播，孤立高点表示局部节点异常。

`plots/top15_relative_l1.png` 的横轴是 relative L1，纵轴是“拓扑序号: 节点名”的 Top-15 tensor。条形越长表示平均幅值偏差越大，最上方最严重。它是优先排查清单，已经按误差排序，所以不能用它判断第一个错误出现在哪里；第一个错误要回到 layerwise 图。

`plots/error_by_op_type.png` 的横轴是算子类别，纵轴是 relative L1。箱体是中间 50% 的 tensor，橙线是中位数，须是正常范围；脚本隐藏了离群点，极端值应查看 CSV。比较箱体高度可以判断问题主要集中在 Conv、MatMul 还是最终输出。

`plots/qdq_scale_distribution.png` 的横轴是 `log10(scale)`，纵轴是 Q/DQ tensor 数量。scale 每增加 1 个 log10 单位，实际量化步长放大 10 倍。它用于观察量化网格是否存在长尾，不能单独作为误差结论；必须和 layerwise error 的拓扑位置结合。

`polygraphy_error_metrics/error_metrics_dets.png` 和 `error_metrics_labels.png` 是最终输出的 Comparator 误差摘要。它们的纵轴是输出误差指标、横轴是对应比较项；这些图只能说明最终输出偏差有多大，不能定位中间层原因，因此应先看 layerwise 图。

快速试跑可以限制前几个 tensor，或跳过 engine 比较：

```bash
python tensort_quant/error_analysis_Conv_MatMul_compare/analyze_quant_error.py \
  --max-tensors 20 --skip-engine
```

`relative_l1` 较大的早期 tensor 是定位重点。若误差首先出现在 DINOv2 attention 的 Q/K/V 或 `Q @ K^T` 输出，随后各层误差持续增长，通常说明 MatMul 激活的 MinMax scale 被离群值拉宽、INT8 分辨率不足；若最早异常出现在 Conv，则应优先检查 Conv 激活校准覆盖和 Q/DQ 插入位置。`layerwise_errors.csv` 给出证据，不能只根据最终 mAP 判断原因。

已在本机 TensorRT 环境用默认输入实际跑过一次。173 个共同输出（171 个 Conv/MatMul 输出加 `dets`、`labels`）中，按拓扑顺序第 5 个输出 `layer.0/attention/MatMul_1` 首次达到 `relative_l1=0.663`、cosine `=0.644`；后续 attention score MatMul 在第 5、7 层达到 `relative_l1=2.333` 和 `2.491`，cosine 分别为负值和约 `0.215`。按算子汇总，MatMul 平均 `relative_l1≈0.993`，Conv 平均约 `0.661`。Q/DQ 审计显示 9/9 Conv 和 154/162 MatMul 的输出被量化，8 个检测头 MatMul 保留 FP32；TensorRT engine 最终 `dets` 的 `relative_l1≈0.697`、`labels≈0.284`。这说明主要故障在 backbone attention 的 MatMul 激活误差累积，不能仅归因于最后检测头。

第二套结果位于 `../error_analysis_Conv_MatMul_compare_smoothquant/`，使用已有的 256 张校准图 SmoothQuant 产物：ModelOpt SmoothQuant、DINOv2 encoder Linear/MatMul INT8，Conv、decoder 和检测头保留 FP32。相同 173 个输出上，MatMul 平均 `relative_l1≈0.290`，Conv 平均约 `0.040`；已有完整 COCO 评测的 mAP@[0.50:0.95] 为 `0.58025`。两套结果的机器可读对比见 `algorithm_comparison.json`。当前 ONNX Runtime 的 Entropy/Percentile 校准在这张全量图上会因校准直方图内存开销被系统 OOM，因此没有把失败的校准过程误写成精度结论。

脚本在 TensorRT 环境缺少 `onnx` wheel 时，会只读地回退到本机 `torch` 环境的同版本 Python site-packages 来做图标记和 Matplotlib 绘图；TensorRT、Polygraphy、ONNX Runtime 和 NumPy 仍来自当前激活的 TensorRT 环境。
