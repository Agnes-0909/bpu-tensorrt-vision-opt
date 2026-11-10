# RF-DETR XLarge 阶段二高级优化方案

## 1. 阶段目标

第一阶段已经把可直接用 SmoothQuant 覆盖的规则矩阵计算做完了，但完整应用的端到端耗时仍然偏高。阶段二的目标不是再把所有节点一律改成 INT8，而是找到计算图中真正占时间的环节，使用 TensorRT 原生融合、定制 attention kernel、选择性低精度和端侧运行时优化，把延迟继续压低，同时把精度损失控制在可部署范围内。

阶段二的优化对象分成三层：

1. **计算图层**：减少 kernel 数、全局内存读写和中间张量落地。
2. **数值层**：只在对精度敏感度可控的矩阵或向量上使用 INT8/FP8，保持 Softmax、归一化和累加的稳定精度。
3. **运行时层**：复用 CUDA buffer 和 stream，使用 CUDA Graph，减少应用层调用开销，使图本身的加速真正反映到端到端延迟。

阶段二的候选必须同时记录 TensorRT profile、应用端 P50/P95、模型精度和中间张量误差。只有单独的 `trtexec` 时间变快而应用端没有变快的候选，不作为最终方案。

## 2. 当前量化版本和测量边界

当前组合是：

| 部分 | 第一阶段方案 | 当前阶段二的处理 |
|---|---|---|
| encoder | ModelOpt SmoothQuant，256 张校准图；主要覆盖 encoder 的 Linear/MatMul，保留精度敏感路径 | 作为 attention 融合和选择性 Q/K/V 低精度的基线 |
| decoder | 当前最优的 decoder SmoothQuant 组合，256 张校准图；主要覆盖 decoder Linear | 暂不把 decoder 当成第一热点，先做轻量融合和运行时验证 |
| head | FP32/原始精度 | 保持不动，避免把极小的计算量换成精度风险 |
| 输入输出 | 输入 FP32，输出 FP32 | 保持接口不变，避免外部转换抵消收益 |

分析边界固定为 batch=1、输入 `1×3×700×700`、同一张 RTX 5060 Ti、同一 TensorRT 11.4。TensorRT profile 使用 warmup 后的 GPU kernel 时间；应用 benchmark 还包括 H2D、D2H、stream 同步、后处理和当前 C++ buffer 管理。

## 3. 已知热点：为什么不能只继续扩大 INT8 范围

### 3.1 阶段分布

来自当前全图 engine 的 TensorRT profile：

| 阶段 | 平均时间 | 占 profile 时间 | 含义 |
|---|---:|---:|---|
| encoder | 14.936 ms | 85.1% | 规则 backbone、12 个 encoder attention block 和 FFN |
| decoder | 2.260 ms | 12.9% | query 数量较少，包含 deformable sampling 和若干小矩阵 |
| head | 0.351 ms | 2.0% | 计算量和可优化空间都很小 |
| 合计 | 17.547 ms | 100% | 仅代表 profile 版本，不等同于应用端完整延迟 |

encoder 的 dense FLOPs 占全图约 95.7%，因此第一阶段的 encoder 量化收益明显。decoder 虽然也能压缩权重，但它在全图中的时间比例和规则 GEMM 比例都小，所以之前 decoder 量化没有形成稳定的端到端收益。

### 3.2 encoder attention 的真实热点

当前 profile 中，每个 encoder block 都出现一组融合的 Softmax 子图、一个 QKᵀ 矩阵乘和一个 AV 矩阵乘。12 个 encoder block 合计约为：

| attention 子环节 | 12 层合计平均时间 | 占全 profile | 说明 |
|---|---:|---:|---|
| `Max → Sub → Exp → Sum → Div` 融合 Softmax | 4.422 ms | 25.2% | 当前最大的单类热点，受归约、指数和中间读写影响 |
| QKᵀ | 2.808 ms | 16.0% | 生成 attention logits，量化误差会改变 token 排序 |
| AV | 3.240 ms | 18.5% | 对 Softmax 权重和 V 做加权累加 |
| QKᵀ + Softmax + AV | 10.470 ms | 59.7% | 阶段二最高优先级 |

这组数据决定了优化方向：**真正值得优先尝试的是把 QKᵀ、稳定 Softmax、AV 融成一个 attention kernel，减少中间张量和 kernel launch；单独给 QKᵀ 或 AV 加 Q/DQ，往往会增加转换和访存，收益不一定为正。**

### 3.3 QKᵀ、Softmax、AV 的数值风险

QKᵀ 输出的是 logits。Softmax 不关心绝对值本身，却非常关心两个相近 logit 的相对大小。一个量化步长造成的微小差异就可能改变最大值附近 token 的排序；排序改变后，后续 AV 的加权组合会沿着 12 个 block 逐步传递。这是它比普通 FFN 更敏感的原因。

Softmax 还包含 `exp`、最大值归约和求和归约。直接把 `exp` 或归约结果做 INT8 会同时面对动态范围和归一化约束，容易出现概率全为零、过度尖锐或和不为 1。推荐的数值路径是：

```text
Q/K/V projection: INT8 或 FP8（保留高质量 scale）
QKᵀ: 低精度乘法，FP16/FP32 累加
scale + mask + stable softmax: FP16（必要时 FP32 累加）
AV: 低精度 V 乘法，FP16/FP32 累加
attention output/residual/LayerNorm: FP16
```

因此“量化 Softmax”与“融合 Softmax”必须分开。阶段二先做后者；真正把 normalization 输出量化为 INT8/FP8，只作为有明确误差收益证据后的实验分支。

## 4. 方案总览和优先级

| 方案 | 主要改动 | 预期收益 | 精度风险 | 工程量 | 阶段二决策 |
|---|---|---:|---:|---:|---|
| A. TensorRT `IAttention`/attention 融合 | 将 QKᵀ、scale/mask、Softmax、AV 改写成可识别 attention 子图，先保持 FP16 | 高，目标是减少 12 组 Softmax 中间 kernel | 低到中，先逐层对齐输出 | 中 | **第一优先级** |
| B. 定制 fused attention plugin | 固定 batch=1、head 数、token 数和 head dim，使用 online Softmax，一次完成 QKᵀ/Softmax/AV | 很高，适合当前固定输入 | 中，需要严格数值校验 | 高 | A 不足时执行 |
| C. 选择性 Q/K/V、AV INT8/FP8 | 低精度乘法，分数和累加保持 FP16/FP32；按 tensor/head 校准 | 中，减少带宽和矩阵乘时间 | 中到高，QKᵀ 会影响排序 | 中到高 | 在 A 后做 |
| D. normalization 输出 FP8/INT8 | 使用 TensorRT attention normalization quantization 接口 | 中，可能减少 attention 输出带宽 | 高，Softmax 概率敏感 | 中 | 后置实验，不作为第一版 |
| E. LayerNorm/residual/pointwise 融合 | 对未被 TensorRT 融合的 Add、LayerNorm、bias、activation 做图模式修正 | 低到中，减少小 kernel 和读写 | 低，数学等价时风险小 | 低到中 | 先检查，已有融合则不改 |
| F. decoder deformable attention 融合 | sampling、weight、value 加权和输出投影合成 plugin | 低到中，decoder 只占约 13% | 中 | 高 | encoder 后再做 |
| G. CUDA Graph + 持久 buffer | 复用 device buffer/stream，捕获固定 shape 的 enqueue | 对应用端可能很高，不改变模型数值 | 很低 | 中 | 与 A 并行 |
| H. TensorRT 构建和 tactic 调优 | timing cache、builder optimization level、aux streams、Tensor Core 对齐 | 低到中，依赖当前 tactic | 很低 | 低 | 每个候选都执行 |
| I. token/分辨率/稀疏 attention | 减少 encoder token 或只计算局部 token | 理论收益最高 | 高，通常需要微调 | 高 | 单独的模型结构路线 |
| J. 结构化 2:4 sparsity/INT4 weight-only | 压缩权重并尝试稀疏 Tensor Core | 可能降低带宽和权重存储 | 中到高，需剪枝/微调 | 高 | 当前不作为首轮 |

优先顺序是 **A → G → C → E → B → F → I/J**。A、G、C 可以分别测量，避免把 kernel、数值和应用调度的问题混在同一轮实验中。

## 5. 方案 A：优先使用 TensorRT 原生 attention 融合

TensorRT 11.4 的 `IAttention` 抽象包含两次 BMM 和一个 normalization，默认会尝试使用单个融合 kernel；只有没有匹配的 kernel 时才建议 decomposable。它的输入文档要求 query/key/value 使用 FLOAT、HALF 或 BF16，因此它首先适合作为 **FP16 attention 融合方案**，不能把“使用 IAttention”直接等同于“Q/K/V 已经是 INT8”。

实施步骤：

1. 用 [`inspect_attention_patterns.py`](inspect_attention_patterns.py) 打印每个 encoder block 的 Q、K、V、Transpose、Mul、Softmax、MatMul、Reshape 节点。
2. 找出 QKᵀ 前后的 `Transpose`、head reshape、scale 和 mask 是否是固定模式。
3. 用 ONNX GraphSurgeon 或 TensorRT Network API 构造 attention 子图，第一版保留 FP16 输入和 FP16 输出。
4. 对照原始 ONNX，比较每个 block attention 输出的 cosine similarity、最大绝对误差和下游 encoder 输出误差。
5. 用 `trtexec --dumpProfile --profilingVerbosity=detailed` 确认 12 个 Softmax 子图是否变成 attention fused layer，以及是否出现额外 reformat。

只有当 profile 证实 Softmax、QKᵀ、AV 的 kernel 数和总时间都下降，才进入下一步低精度实验。若输入布局导致额外 transpose/reformat，融合可能反而变慢，必须以 profile 为准。

本次只读检查已经确认了改写入口：该 ONNX 有 12 个 encoder attention block；每个 block 都包含显式的 `MatMul → Softmax → MatMul_1`，前面是 query/key/value 的 reshape、transpose、scale，后面接 attention output projection。SmoothQuant 已在 Q/K/V projection 周围插入 Q/DQ，但 QKᵀ、Softmax、AV 仍以独立 ONNX 节点存在。这说明下一轮可以保持现有 QKV 量化范围，只把中间 attention 子图改写成 fused FP16 attention，实验变量是清楚的。

## 6. 方案 B：定制 fused attention plugin

如果 ONNX 模式无法稳定触发 `IAttention`，可以为当前固定形状编写 plugin。当前模型的有利条件是 batch=1、encoder token 数固定、head 数和 head dim 固定，可以针对实际尺寸选择 tile 和 shared-memory 布局。

推荐的 kernel 逻辑是在线 Softmax：

```text
对每个 query tile：
  计算 Q·Kᵀ，并在 FP16/FP32 中维护 running_max 和 running_sum
  立即完成 exp/归一化
  直接读取 V 并累加 attention·V
  只写出最终 attention 输出，不写出完整 logits/probability 矩阵
```

这会同时减少全局内存流量、Softmax 中间 tensor 和 kernel launch。plugin 第一版应保持 Q/K/V 和累加 FP16，先只验证融合收益；第二版再把 Q/K/V 或 V 的输入换成 INT8/FP8，并保留 score、Softmax 和累加的 FP16/FP32。

plugin 的主要风险是：head layout、mask 语义、scale 位置和 residual 输入稍有不一致就会造成检测结果漂移。必须用 Polygraphy 抓取同一输入的中间结果，并对每一层 attention 输出做误差门禁。

本轮已完成固定形状参考实现、PyTorch SDPA/FlashAttention 原型和独立 CUTLASS plugin，代码和复现命令见 [`experiments/fused_attention_plugin/README.md`](experiments/fused_attention_plugin/README.md)。tiled CUDA kernel 约 108 ms，保留为正确性诊断基线；SDPA/FlashAttention FP16 attention 约 8.03 ms，相对 baseline 16.69 ms 约降低 51.9%，但依赖 libtorch。独立 CUTLASS plugin 约 9.99 ms，完整 COCO mAP 为 0.5806461，并已验证使用 `--setPluginsToSerialize` 嵌入 engine 后可以去除运行时 libtorch 和外部 plugin 查找。当前仍固定 `[B,H,S,D]` layout、`B=1,H=12,S=1226,D=64`，变长序列和其它 head 配置需要单独扩展。

## 7. 方案 C：选择性 Q/K/V、QKᵀ 和 AV 低精度

### 7.1 推荐的试验矩阵

| 实验 | Q/K/V projection | QKᵀ | Softmax | AV | 累加 | 目的 |
|---|---|---|---|---|---|---|
| C0 | 当前 SmoothQuant | FP16 | FP16 | FP16 | FP16 | attention 融合后的参考 |
| C1 | INT8 | INT8 输入乘法 | FP16 | FP16 | FP16 | 先测 QKᵀ 的 INT8 Tensor Core 是否有收益 |
| C2 | INT8 | INT8 输入乘法 | FP16 | INT8 V 乘法 | FP16 | 测 AV 带宽和乘法收益 |
| C3 | FP8 | FP8 输入乘法 | FP16 | FP8 V 乘法 | FP16 | 在支持 FP8 Tensor Core 的 GPU 上测吞吐/精度折中 |
| C4 | INT8/FP8 | INT8/FP8 | INT8/FP8 | INT8/FP8 | FP16 | 只作为极限实验，不作为默认部署路径 |

C1/C2 的 scale 必须使用校准集统计，而不是用 QKV projection 的权重 scale 直接复用。至少保存 per-tensor 的 min/max、percentile、绝对最大值和饱和比例；若单一 tensor scale 造成明显 outlier，可进一步试 per-head 或 SmoothQuant 风格的通道迁移。TensorRT 原生 activation quantization 不能简单地假设任意轴都支持 per-channel，要以实际 layer/API 约束为准。

### 7.2 为什么不直接量化 Softmax

Softmax 输入 logits 的范围随层和图像变化，且最大值减法、指数和归一化是一个耦合过程。直接插入 `QuantizeLinear → Softmax → DequantizeLinear` 会产生三个问题：

- logits 接近时，INT8 步长可能改变最大值附近的排序；
- 概率很小的 token 可能全部被压成 0，导致 AV 的有效上下文缩窄；
- Q/DQ 的额外 kernel 和 layout reformat 可能抵消省下的算术时间。

更稳妥的做法是使用 fused stable Softmax，内部使用 FP16/FP32 累加；如果要试 normalization 输出量化，优先试 FP8，并将其作为 attention output 的独立消融，而不是同时量化 logits、概率和 AV。

## 8. 方案 E：LayerNorm、residual 和 pointwise 融合

当前 profile 已经出现很长的 TensorRT 融合层名，例如 `__myl_TranConcAddReshMeanSubMulMean...`，说明至少部分 reshape、残差、均值、方差和 scale 已被 TensorRT 合并。手工把整个 LayerNorm/residual 重新写一遍不一定更快，可能由于布局不匹配插入 reformat。

检查规则如下：

1. 先在 `--profilingVerbosity=detailed` 的 profile 中确认每个 Add、LayerNorm、bias、activation 是否独立存在。
2. 只有独立存在且占用明显时间的模式才做 ONNX rewrite。
3. rewrite 只做数学等价的模式整理：`bias + activation`、`residual + normalization`、连续 pointwise 合并；不改变归一化 epsilon、broadcast 轴和计算顺序。
4. 每次 rewrite 都要比较输出误差、reformat 数量和 profile 时间。

这条路线是低风险的补充优化，预期收益小于 attention 融合。它不能替代 attention kernel，因为当前 Softmax/QKᵀ/AV 才是 59.7% 的 profile 时间。

## 9. 方案 G：端侧运行时优化

当前 [`tensorrt_infer/src/infer.cpp`](../tensorrt_infer/src/infer.cpp) 的 `infer()` 每次调用都会创建 stream、`cudaMalloc` 输入、为每个输出 `cudaMalloc`，完成同步后再释放。benchmark 也明确说明 inference 时间包含 H2D/D2H 和每调用一次的 CUDA allocation。这会让单次端侧延迟看起来远高于纯 TensorRT kernel 时间。

应按下面顺序改造：

1. 在 `TensorRTEngine` 构造或 `prepare()` 阶段分配固定 shape 的输入/输出 device buffer。
2. 复用一个 stream，不在每次推理时 create/destroy。
3. 使用 pinned host memory，异步 H2D、enqueueV3、D2H，并只在需要读结果时同步。
4. 固定输入 shape 后捕获 CUDA Graph；预热、地址绑定和输出 buffer 都固定后再 capture。
5. 分别报告 `enqueueV3` GPU 时间、H2D/D2H、应用端完整 inference 和 preprocess/postprocess，避免把不同指标混为一个数字。

这一项不改变模型数值，通常比继续扩大 decoder INT8 范围更可能改善应用端 P50/P95。CUDA Graph 是否能捕获当前 enqueue 路径要用实际 TensorRT context 验证；若动态 shape、输出 allocator 或 host 回调阻止 capture，需要先固定 profile 和输出地址。

## 10. 其他候选路线

### 10.1 decoder deformable attention

decoder 只占约 12.9% 的 profile 时间。它的主要问题不是大矩阵，而是 sampling、GridSample、Gather、权重乘法和残差归一化的 kernel 数。可以在 encoder attention 之后尝试把 value projection、采样和 weighted sum 合成 plugin，但要先确认当前 TensorRT 是否已经把 `sampling_offsets/attention_weights/MatMul` 部分合并。它是二阶段后半段的优化，不是当前最高 ROI。

### 10.2 token 数量和输入分辨率

减少 encoder token、降低输入分辨率、局部窗口 attention 或 token pruning 会同时减少 QKᵀ、Softmax、AV 和 FFN 的计算量，理论收益超过单纯量化。但这改变模型结构或有效输入信息，通常需要微调和重新评估 COCO。应作为独立的模型结构项目，不能与无训练的 TensorRT 优化混在一个候选中。

### 10.3 FP8、INT4 和结构化稀疏

TensorRT 11.4 API 支持 FP8 数据类型和量化层，适合在支持对应 Tensor Core 的设备上做选择性 FP8 实验。FP8 比 INT8 更适合保留动态范围，但仍需要对 Q/K logits 和 FFN outlier 做校准。INT4 weight-only 更偏向降低权重带宽和存储，在当前 batch=1、encoder 计算密集的场景中不保证自动转化为更短延迟；2:4 structured sparsity 也需要剪枝/微调并确认实际 tactic 命中。两者放到 attention 融合和运行时优化之后。

## 11. 推荐的实验批次

### S0：重新建立可比较基线

固定同一 engine、同一输入和同一 GPU，分别记录：

- `trtexec` 纯 enqueue 的 mean/P50/P95；
- TensorRT detailed profile 的 layer 时间；
- 当前 C++ 应用的 inference mean/P50/P95；
- H2D、D2H、preprocess、postprocess 和 CUDA allocation 的时间。

所有后续候选均使用同一 warmup 和迭代数，profile 版和非 profile 版分开，避免 detailed profiling 本身改变延迟。

### S1：确认 TensorRT 已有融合

执行：

```bash
cd /home/rton/rfdetr/stage2_advanced_opt
python profile_hotspots.py \
  --profile-csv ../tensorrt_quant_decoder/compute_graph_analysis/tensorrt_profile_layers.csv \
  --out-dir analysis_outputs

conda run -n TensorRT python inspect_attention_patterns.py \
  ../tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx
```

输出的 `hotspot_summary.csv/json` 和两张 PNG 用于后续每个候选的对照。attention pattern 检查脚本是只读的，不会改写 ONNX。

### S2：FP16 attention fusion

先只改写 attention，保持其它量化范围不变。验收重点是 Softmax 融合层数量、QKᵀ/AV 总时间、reformat 数量和 COCO mAP。若 fused kernel 没有命中，记录具体 layout/shape 原因，转入 plugin 路线。

### S3：运行时 buffer/CUDA Graph

在 S2 的最佳 engine 上改造 C++ 推理类，并分别比较：

```text
旧 infer()（每次 malloc/free）
持久 buffer + 持久 stream
持久 buffer + CUDA Graph
```

这三组必须使用同一 engine，才能看出端侧调度开销本身的贡献。

### S4：选择性低精度 attention

按 C1、C2、C3 的矩阵逐步扩大范围；任何一次出现明显 mAP 或 attention 输出误差异常，都回退上一步，而不是继续叠加其它低精度选项。

### S5：decoder plugin 和结构路线

只有 encoder attention 和运行时优化完成后，才评估 decoder deformable attention、token pruning、分辨率变化、sparsity 或 INT4。它们需要单独的精度/训练决策。

## 12. 精度和性能验收门禁

建议把部署候选的硬门禁固定为：

| 指标 | 门禁 |
|---|---|
| COCO 全量验证集 mAP | 相对当前组合绝对下降不超过 0.0015；如果产品有更严格指标，按产品指标替换 |
| 关键 attention 中间张量 | 无 NaN/Inf；cosine similarity ≥ 0.999；相对 L1 变化按层记录，不能只看最终输出 |
| 检测输出 | box/class 的数量、score 分布和高置信目标不出现系统性漂移 |
| 端到端延迟 | 同一 benchmark 条件下 P50、P95 同时改善；只改善 trtexec 不算通过 |
| TensorRT profile | 融合后 kernel 数、reformat 数和热点总时间都可解释；不能以新增大 kernel 掩盖小 kernel |
| 可复现性 | 保存构图脚本、TensorRT 版本、GPU、输入 shape、校准集和完整命令 |

## 13. 当前结论

第一步应该是 **FP16 attention 融合 + 持久 buffer/CUDA Graph**。它们分别针对 profile 的最大计算热点和当前应用端的调度开销，精度风险最低。QKᵀ、AV 的 INT8/FP8 应放在融合之后，并保留 FP16/FP32 的 score、Softmax 和累加；直接量化 Softmax 是高风险、低确定性的后置实验。LayerNorm/residual 需要先检查 TensorRT 是否已经融合，当前 profile 已经显示部分融合，不能先假定它是独立瓶颈。

## 14. 参考资料和本机依据

- TensorRT Performance Best Practices（attention fusion、layer fusion、CUDA Graph、Q/DQ fusion、timing cache）：[NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- TensorRT Model Optimizer：[NVIDIA/TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- 本机 TensorRT 11.4 C++ API：[`NvInfer.h`](../../TensorRT/TensorRT-11.4.0.106/include/NvInfer.h)，其中 `IAttention` 描述了 BMM1、normalization、BMM2 的融合模型，`IQuantizeLayer`/`IDequantizeLayer` 描述 FP8/INT8 支持和 activation per-tensor 约束。
- 当前全图计算量和 profile：[`COMPUTE_DISTRIBUTION_REPORT.md`](../tensorrt_quant_decoder/compute_graph_analysis/COMPUTE_DISTRIBUTION_REPORT.md)
- 当前端侧实现：[`infer.cpp`](../tensorrt_infer/src/infer.cpp)、[`benchmark.cpp`](../tensorrt_infer/src/benchmark.cpp)
