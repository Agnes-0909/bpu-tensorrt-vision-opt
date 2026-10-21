# 项目报告一：ViT/DETR 跨平台视觉大模型部署优化

仓库名：`bpu-tensorrt-vision-opt`

建议仓库描述：

> Deep deployment optimization for ViT/DETR on Horizon BPU and NVIDIA TensorRT/CUDA, including graph rewriting, quantization diagnosis, operator fusion, custom operators, and high-performance kernels.

版本：v1.0  
主平台：地平线 X5  
辅助平台：地平线 S100E、NVIDIA GeForce RTX 5060 Ti

> 文档中的“起始参考范围”用于安排实验，不能替代实测结果。正式 baseline 必须在固定模型、输入、软件版本和功耗模式下冻结。

## 项目目标：ViT/DETR 视觉大模型跨平台深度部署优化

### 1. 项目目标及 baseline

#### 1.1 项目名称

**面向地平线 BPU 与 NVIDIA GPU 的 ViT/DETR 视觉大模型编译、量化与算子优化系统**

#### 1.2 具体应用场景

选择一个真实的机器人或智能驾驶视觉任务，统一使用以下模型进行部署：

- **分类/特征模型：ViT-B/16**，用于目标识别、图像特征提取或 ReID；
- **检测模型：RT-DETR-S 或 RT-DETR-R18**，用于行人、车辆和障碍物检测；
- 可选扩展：将检测模型替换为 DINO/Deformable DETR 的轻量版本，专门研究 Multi-Scale Deformable Attention 的部署。

建议第一阶段固定为 `ViT-B/16 + RT-DETR-R18`，不要同时引入过多模型。项目的重点是解决深层部署问题，而不是扩大模型数量。

典型输入：

- ViT：`1 x 3 x 224 x 224`；
- RT-DETR：`1 x 3 x 640 x 640`；
- 额外测试动态分辨率：`480/640/800` 或 `224/384`。

目标是完成从 PyTorch/ONNX 到地平线 BPU 和 TensorRT 的完整部署，并形成可解释的优化闭环：

```text
模型导出
→ 图结构分析
→ 算子支持排查
→ 图修改与算子融合
→ 量化校准
→ 自定义算子
→ 编译
→ 运行时 profiling
→ 精度回归
→ 版本化交付
```

#### 1.3 平台分工

| 平台 | 主要用途 | 必须掌握的深层能力 |
|---|---|---|
| 地平线 X5 | 主平台，完成 BPU 部署和端到端优化 | BPU 图转换、定点量化、图切分、输入输出内存、算子替换、板端 profiling |
| 地平线 S100E | 辅助平台，用于验证不同地平线硬件/工具链差异 | 编译结果差异、算子支持差异、量化差异、Runtime 适配 |
| NVIDIA RTX 5060 Ti | GPU 参考平台和高性能算子开发平台 | TensorRT、CUDA、Plugin、TileLang/Triton 计算图优化、Nsight 诊断 |

“TileLong”按本项目暂按 **TileLang/Triton 类 tile 化计算图和高性能算子优化**理解；如果实际工具名称不同，在立项时锁定具体版本和仓库。

#### 1.4 Baseline 冻结规则

正式 baseline 必须同时记录：

- 模型 commit 和权重 SHA256；
- PyTorch、ONNX、TensorRT、地平线 SDK 版本；
- 编译参数、输入 shape、batch size；
- CPU/GPU/BPU 频率和功耗模式；
- warmup 次数、测量次数、p50/p95/p99；
- 端到端延迟和纯模型延迟；
- 峰值内存、模型大小、功耗；
- 精度指标和数据集版本。

建议冻结四个起始 baseline：

| 模型 | 平台 | 起始 baseline 配置 | 建议记录的参考范围 |
|---|---|---|---|
| ViT-B/16 | X5 | 浮点或平台默认高精度配置 | 单帧端到端约 40～100 ms；必须以实测冻结 |
| ViT-B/16 | RTX 5060 Ti | TensorRT FP16，batch=1 | 通常低于 X5，记录纯推理与预处理分项 |
| RT-DETR-R18/S | X5 | INT8 或平台推荐定点配置 | 单帧端到端约 30～80 ms；重点看后处理和内存拷贝 |
| RT-DETR-R18/S | RTX 5060 Ti | TensorRT FP16/INT8，batch=1 | 记录 640² 输入下的 p50/p95/p99 |

S100E 不要求与 X5 直接比较绝对延迟，而是用于记录：

- 同一 ONNX 图在不同地平线平台上的落图比例；
- 同一量化参数的精度差异；
- 同一算子在不同 BPU/SDK 版本上的支持情况；
- 模型编译时间、模型大小和运行时内存差异。

建议将第一版正式 baseline 保存为：

```text
baseline/
├── model_manifest.yaml
├── x5_v1.json
├── s100e_v1.json
├── tensorrt_5060ti_v1.json
├── accuracy_report.json
└── profiling_traces/
```

#### 1.5 项目目标

完成后应达到以下结果：

1. ViT 和 RT-DETR 均能在 X5、S100E 和 RTX 5060 Ti 上运行；
2. 能自动定位模型无法编译、算子未落后端、精度异常和延迟异常的原因；
3. 至少完成一套图修改方案、一套量化优化方案和一套自定义算子方案；
4. TensorRT 侧至少完成一个 CUDA 高性能算子和一个 TensorRT Plugin；
5. 能给出不同平台下的性能/精度/内存 Pareto 结果；
6. 所有优化都保留优化前后可复现的基准和回归测试。

### 2. 重点完成项目列表

#### 2.1 模型导出与图结构审计

- 从 PyTorch 导出 ONNX，并验证数值一致性；
- 检查动态 shape、轴顺序、广播、常量折叠和数据类型；
- 对 ViT 重点检查：Patch Embedding、QKV、Transpose、Reshape、LayerNorm、GELU、Softmax；
- 对 RT-DETR 重点检查：Multi-Scale Feature、Transformer Encoder/Decoder、Attention、TopK、box decode、NMS；
- 生成每个后端的算子覆盖矩阵；
- 记录每个算子所属子图、输入输出 shape、dtype 和内存布局。

交付物：

```text
operator_matrix.csv
model_graph_before.onnx
model_graph_annotated.html
unsupported_ops.md
```

#### 2.2 地平线 BPU 图修改与算子融合

至少完成以下图修改实验：

- `Conv + BatchNorm + Activation` 融合；
- LayerNorm 结构改写或替换；
- GELU 改写为平台可支持的近似形式；
- QKV 三个 Linear 合并或转换为适合后端的形式；
- Multi-Head Attention 的 Transpose/Reshape 合并；
- RT-DETR 中多尺度特征拼接和 Resize 的结构重写；
- 将 DFL decode、box decode 和部分后处理从 Python 移出；
- 比较 BPU、CPU fallback 和自定义算子的代价。

每个改动必须包含：

```text
原始子图
修改后子图
精度变化
编译结果
BPU 落图比例
延迟变化
```

#### 2.3 地平线量化排查与优化

建立完整的量化诊断流程：

- FP32、FP16、INT8 三套结果逐层对齐；
- 统计各层 activation min/max、均值、方差和异常值比例；
- 对 ViT 的 QKV、Attention score、LayerNorm 进行敏感度分析；
- 对 RT-DETR 的小目标检测头、分类头、box regression head 进行敏感度分析；
- 比较随机校准、分层校准和困难样本校准；
- 识别量化后召回率下降来自权重、激活、输入预处理还是后处理；
- 对敏感层使用混合精度或保留高精度；
- 建立校准集版本化和量化参数版本化机制。

必须完成一个“量化异常自动报告”：

```text
layer_name
activation_range
saturation_ratio
cosine_similarity
accuracy_impact
recommended_precision
```

#### 2.4 地平线自定义算子或 fallback 算子

选择一个具有实际收益的算子进行实现：

- RT-DETR box decode；
- DFL decode；
- Attention 中的特定融合子图；
- ViT 的 LayerNorm/GELU 融合；
- 检测后处理中的阈值、TopK 或 NMS。

要求完成三种版本对比：

1. 原始框架实现；
2. 后端支持算子重写版本；
3. 地平线 custom-op 或 C++ CPU fallback 版本。

需要证明：

- custom-op 是否真正减少了数据拷贝；
- custom-op 是否增加了编译和维护成本；
- 如果算子不能落 BPU，CPU fallback 是否仍然值得采用；
- 该算子在 X5 和 S100E 上是否具有相同表现。

#### 2.5 TensorRT 图优化与 Plugin

TensorRT 侧至少完成：

- FP16 和 INT8 engine 构建；
- static shape 和 dynamic shape 对比；
- layer fusion、tactic、workspace、engine inspector 分析；
- 为 RT-DETR 实现 `box decode + sigmoid + topK/NMS` 融合 Plugin；
- 为 ViT 实现 `QKV projection + reshape/transpose` 或 `LayerNorm + residual` 融合方案；
- 比较 TensorRT 原生实现、Plugin 和 CUDA 独立 kernel。

优先使用当前 TensorRT 的 `IPluginV3`；如果 SDK 版本限制只能使用旧 API，则保留旧版实现并说明迁移差异。

#### 2.6 CUDA/Triton/TileLang 高性能算子

至少实现以下两类算子：

- CUDA C++ 版本；
- Triton 或 TileLang 版本；
- TensorRT Plugin 调用版本。

建议目标：

1. `QKV projection + transpose + split`；
2. `LayerNorm + residual + activation`；
3. `RT-DETR box decode + NMS`；
4. ViT patch embedding 的 fused projection；
5. 预处理中的 resize、normalize 和 layout transform。

必须使用 Nsight Systems/Compute 分析：

- global memory throughput；
- shared memory bank conflict；
- register pressure；
- occupancy；
- kernel launch 数量；
- CPU/GPU 同步；
- stream overlap；
- CUDA Graph 是否有效。

#### 2.7 问题诊断系统

形成一套可复用的诊断流程：

| 问题 | 必须能回答 |
|---|---|
| 编译失败 | 是算子不支持、shape 不合法、dtype 不匹配还是内存限制？ |
| 延迟过高 | 是计算慢、内存拷贝、kernel launch、同步还是后处理？ |
| INT8 精度下降 | 是校准集、activation saturation、敏感层还是预处理差异？ |
| BPU 落图率低 | 哪些子图落 CPU，代价是多少，能否通过图修改解决？ |
| TensorRT engine 不稳定 | tactic、workspace、动态 shape 还是多 stream 调度导致？ |
| 输出不一致 | 是布局、量化 scale、坐标变换还是后处理实现差异？ |

### 3. 验收标准

#### 3.1 功能验收

- ViT-B/16 和 RT-DETR 至少各完成一套 X5、S100E、TensorRT 部署产物；
- 支持 FP32/FP16/INT8 中至少两种精度；
- 提供 C++ 或 Python 推理接口和命令行 benchmark；
- 支持固定输入和至少一种动态输入配置；
- 所有后端均有自动化精度回归。

#### 3.2 精度验收

- FP32 与部署前向结果最大误差有明确阈值；
- INT8 相对 FP32 的主任务指标下降不超过 1～2 个百分点；
- 对 ViT 特征提取，Top-1 或 embedding cosine similarity 满足预设阈值；
- 对 RT-DETR，mAP、召回率和小目标召回率分别报告；
- 所有精度下降都能定位到具体层、具体输入或具体后处理步骤。

#### 3.3 性能验收

- 所有结果同时报告 p50、p95、p99，不能只报告平均 FPS；
- X5 上端到端延迟相对初始 baseline 至少提升 25%；
- TensorRT/5060 Ti 上纯模型或端到端延迟至少提升 30%；
- 自定义 CUDA/Plugin 算子相对原始 Python/CPU 实现至少提升 2 倍；
- 融合后的 kernel 数量或中间 tensor 读写量有可量化下降；
- 端到端 profiling 能解释至少 90% 的总延迟。

#### 3.4 工程验收

- 提供一键导出、转换、编译、运行和 benchmark 脚本；
- 所有模型、SDK、权重、校准集和编译参数可追溯；
- 至少包含 5 个编译失败案例、5 个精度异常案例和 5 个性能优化案例；
- 输出最终性能/精度/内存 Pareto 报告；
- 形成一篇不少于 15 页的技术报告。

---

## 仓库交付结构

```text
bpu-tensorrt-vision-opt/
├── models/
├── export/
├── convert/
├── custom_ops/
├── runtime/
├── profiling/
├── accuracy/
├── benchmarks/
├── scripts/
├── configs/
├── Dockerfile
└── README.md
```

README 必须包含：环境版本、硬件信息、模型权重来源、运行命令、baseline、优化前后对比、已知限制和复现实验步骤。

## 面试展示重点

不要只展示最终 FPS。需要展示“发现问题 → 设计实验 → 定位根因 → 修改模型图或 Runtime → 实现自定义算子 → 精度回归 → 用 profiler 证明收益”的完整闭环。
