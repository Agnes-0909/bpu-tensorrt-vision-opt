# 阶段三：前处理与后处理 CUDA 加速方案

## 1. 阶段目标

阶段二已经把主要模型计算热点定位到 encoder attention，并完成了一个
FlashAttention plugin 原型。阶段三的目标是把端侧输入处理和输出解码纳入同一条
GPU pipeline，减少 CPU 计算、host/device 往返和每帧同步，同时保持与当前 RF-DETR
预处理、后处理和 COCO 评测完全一致。

阶段三与模型侧优化并行推进。PyTorch SDPA FlashAttention 仍作为性能上限对照；不依赖
libtorch 的独立 CUTLASS plugin 已完成并通过完整 COCO 与 C++ runtime 验证，且已验证
可以把 plugin 嵌入 engine。LayerNorm/residual 已完成 profile 审计，QK/AV INT8 已完成
精度消融并因超门槛而拒绝；变长 shape、FP8 和目标端兼容性仍需单独验证。前处理实验
与这些模型 kernel 实验分开测量，最终部署时优先组合已通过门禁的候选。

本阶段优先优化前处理。当前 C++ benchmark 中前处理约 6.78 ms、模型 inference
约 8.77 ms、后处理约 0.07 ms；因此前处理是当前端到端链路中更值得投入的部分。
P1 已完成第一版验证：在 RTX 5060 Ti 上，CUDA 前处理（RGB uint8 H2D 加 fused
resize/normalize/layout）约 0.21–0.35 ms。
Python COCO evaluator 中前处理约 30.5 ms，其中还包含 Python 图像读取和计时开销，
不能直接与 C++ benchmark 的 6.78 ms 混为一谈。所有阶段三结论以 C++ persistent
runtime 的分段测量为主。

## 2. 当前端侧基线和正确性约束

当前输入路径是：

```text
OpenCV imread(BGR, uint8)
  -> cvtColor(BGR -> RGB)
  -> CPU bilinear resize
  -> /255
  -> ImageNet normalize
  -> HWC -> NCHW FP32
  -> H2D cudaMemcpyAsync
  -> TensorRT enqueueV3
  -> D2H outputs
  -> CPU sigmoid / argmax / box decode / threshold
```

C++ 基线实现位于 [`../tensorrt_infer/src/preprocess.cpp`](../tensorrt_infer/src/preprocess.cpp)。
预处理必须保持以下语义：

- 输入颜色顺序为 RGB，源像素为 interleaved `uint8`；
- resize 使用 torchvision 风格的 `align_corners=False` 双线性规则，antialias 关闭；
- 源坐标公式为 `(dst + 0.5) * src / dst - 0.5`，边界坐标 clamp 到合法范围；
- 归一化为 `(pixel / 255 - mean) / std`；
- 输出布局为连续的 `[1, 3, 700, 700]` FP32 NCHW；
- 后处理保持当前类别槽位、背景类排除、sigmoid、box decode 和阈值行为。

任何 CUDA 版本都必须先通过同一张图片和多张代表性图片的数值对比，再进行 COCO
评测。不能因为输出 mAP 接近就跳过 resize 边界和颜色通道校验。

## 3. 方案分层

### P0：分段基线和数据流确认

先给现有 C++ 路径增加独立计时：

```text
image decode
BGR -> RGB
CPU resize
CPU normalize + layout
H2D
TensorRT inference
D2H
CPU postprocess
```

每个阶段记录 mean、P50、P95、P99、最大值；GPU 阶段使用 CUDA event，CPU 阶段
使用 steady clock。P0 的结果作为所有 CUDA 版本的同机对照，也要区分“文件 JPEG 解码”
和“已经得到 RGB uint8 后的张量预处理”。

### P1：CPU 解码 + CUDA fused preprocess（第一版已完成）

第一版只改变 resize、颜色/布局和归一化，不引入 GPU JPEG 解码，以便把变量控制在
CUDA 前处理 kernel：

```text
OpenCV decode BGR uint8
  -> CPU 或上传前完成 RGB 转换
  -> H2D uint8 RGB
  -> CUDA fused resize + normalize + HWC->NCHW
  -> 直接写 TensorRT input device buffer
```

当前实现位于 `cuda_preprocess.cu`：每个输出像素对应一个 thread，thread 内计算四个
邻域像素和三个通道；均值、标准差使用 kernel 常量，FP32 计算并直接写入连续 NCHW
输入。它复现 torchvision 风格的 `align_corners=False`、antialias 关闭的双线性规则。
后续再尝试向量化读写、shared memory tile 和 half 输入。

P1 的关键收益是合并 CPU resize、normalize、layout 三次遍历，并避免先生成完整的
host FP32 NCHW 张量。模型 input device buffer 应由 persistent runtime 提供给前处理
kernel，前处理完成后直接在同一 CUDA stream 上调用 `enqueueV3`。

### P2：pinned memory 和异步流水

在 P1 正确后，使用 page-locked host memory 保存解码后的 RGB 图像，并让上传、CUDA
前处理和 TensorRT enqueue 使用同一条持久 stream。对于连续帧输入，可以使用双缓冲：

```text
buffer 0: H2D -> preprocess -> inference
buffer 1: CPU/GPU decode
```

P2 的测量必须分别报告 kernel 时间和端到端时间，避免把异步 overlap 的收益误归为
resize kernel 加速。单张离线 JPEG 如果没有下一帧可以并行，双缓冲收益可能很小。

当前已先验证 P2 的第一步：`cuda_preprocess_async` 不在 CUDA 前处理和 TensorRT
enqueue 之间插入主机同步，前处理和推理在同一条 persistent stream 上顺序执行。FlashAttention
FP16 engine 单图测试中，带显式阶段同步的总时间约 `8.35 ms`，异步串接约 `8.23 ms`，
收益约 `1.4%`。这只是去掉中间同步的结果，尚未包含 pinned host buffer 和双缓冲；后续
需要在连续帧输入下测量 overlap。

P2 的 pinned host buffer 第一版也已实现。前处理 benchmark 中 pageable RGB H2D 约
`0.394 ms`，pinned RGB H2D 约 `0.246 ms`，输入误差仍为 `8.34e-7`；FlashAttention
端到端的 pinned 前处理约 `0.151 ms`。当前 benchmark 在每轮测试前把已解码的 RGB
复制到 pinned buffer，这个一次性 CPU copy 不计入 steady-state；生产流水需要让解码器
直接输出到 pinned memory，才能保留完整收益。

固定输入 shape 的 CUDA Graph capture/replay 也已完成。FlashAttention engine 的推理段
从约 `8.10 ms` 降到 `7.21 ms`；完整 COCO 5000 张 mAP 为 `0.580853`，与 CUDA P1
路径一致。Graph 目前只捕获 TensorRT enqueue，输出 D2H 仍在 graph 外完成；动态 shape
和多 context 并发尚未验证。

### P3：GPU JPEG/视频解码

nvJPEG 第一版原型已经实现并测试：JPEG 直接解码到 GPU RGBI buffer，再调用 device-source
版本的 fused preprocess。单图 640×428→700×700 平均 `0.747 ms`，输入 max absolute
error `8.34e-7`。该耗时包含 nvJPEG 解码和 CUDA resize/normalize/layout，不包含文件读取；
完整端到端收益仍需在连续帧和目标端验证。

当 P1/P2 的瓶颈转移到图像读取后，再尝试：

- JPEG 图片：`nvJPEG` 输出 GPU RGB buffer；
- 摄像头或视频：优先考虑 NVDEC 或已有 GPU surface；
- 尽量让解码输出直接进入 P1 的 CUDA preprocess 输入，不回到 CPU。

P3 对静态图片的收益取决于 JPEG 解码占比和文件 I/O；如果端侧输入已经是摄像头
GPU surface，P3 的收益通常比离线文件更明显。P3 需要单独记录 CPU 解码时间、GPU 解码
时间和文件读取时间。

### P4：CUDA 后处理（低优先级）

当前 C++ 后处理约 0.07 ms，因此只在 P1～P3 完成后验证。候选 kernel 可以融合：

```text
labels sigmoid
  -> 每个 query 的类别 argmax
  -> score threshold
  -> cxcywh -> xyxy / 图像坐标
  -> clamp
  -> compact selected detections
```

如果后处理放到 GPU，TensorRT 输出应暂留在 device buffer，CUDA kernel 只把筛选后的
检测结果拷回 CPU，避免先拷回全部 `dets` 和 `labels`。由于当前只有 300 个 query，必须
用实测证明 kernel launch、compact 和同步开销小于 CPU 的 0.07 ms，不能假定 GPU 一定更快。

## 4. 推荐实现接口

当前已新增独立的 CUDA 前处理模块，不把 kernel 直接塞进 TensorRT plugin：

```text
stage3_accelerate_process/
  README.md
  STAGE3_PREPROCESS_OPTIMIZATION_PLAN.md
  cuda_preprocess.cu/.h       # 已实现 resize + normalize + NCHW
  cuda_postprocess.cu/.h      # 后续可选
  benchmark_preprocess.cpp    # 已实现 CPU/CUDA 数值和吞吐 benchmark
  benchmark_nvjpeg_preprocess.cpp # nvJPEG GPU 解码 + fused preprocess benchmark
  ../tensorrt_infer/src/benchmark.cpp # 五种 runtime 模式的端到端分段 benchmark
  evaluate_coco_cuda_preprocess.cpp # CPU/CUDA 前处理 COCO prediction 生成器
  evaluate_coco_preprocess.py # pycocotools mAP 对比脚本
  experiments/                 # 本地产物，默认不进 Git
```

`TensorRTEngine` 已增加一条 device-input API：

```cpp
infer_persistent_device_input()
```

或者把 CUDA preprocess context 作为 `TensorRTEngine` 的组成部分，使同一条 persistent
stream 和 input buffer 被复用。当前 `infer_persistent()` 仍接收 host `std::vector<float>`
作为 CPU 对照；`infer_persistent_device_input()` 直接消费 CUDA preprocess 写入的
`persistent_input_device_`，删除了“host FP32 tensor -> H2D”的中间步骤。

## 5. 实验矩阵

| 方案 | JPEG 解码 | resize/normalize/layout | H2D | 后处理 | 目的 |
|---|---|---|---|---|---|
| P0 CPU baseline | OpenCV CPU | CPU | 普通 host | CPU | 建立分段基线 |
| P1 fused CUDA | OpenCV CPU | CUDA fused | uint8 RGB 上传 | CPU | 验证 kernel 和直接写 input buffer |
| P2 async | OpenCV CPU | CUDA fused | pinned + async | CPU | 测 overlap 和 host/device 管理收益 |
| P3 GPU decode | nvJPEG/NVDEC | CUDA fused | 尽量零拷贝 | CPU | 消除 CPU decode 和往返 |
| P4 full GPU post | nvJPEG/NVDEC 或 P2 | CUDA fused | 设备内 | CUDA | 仅在后处理成为新热点时验证 |

每个方案固定同一 engine、同一输入尺寸、同一 GPU 和同一图片集合，至少报告：

- decode、preprocess、H2D、inference、D2H、postprocess 的 mean/P95/P99；
- C++ 单图 steady-state 延迟和连续帧吞吐；
- GPU kernel 时间、CPU wall time 和同步等待时间；
- preprocess 输出的 max absolute error、RMSE、relative L1、cosine；
- 完整输出的 `dets`/`labels` 误差；
- COCO mAP@[.5:.95]、mAP50、mAP75；
- 不同图片尺寸、亮度、宽高比和目标大小下的稳定性。

## 6. 验收门槛

### P1 实测结果

在 4 张不同尺寸的 COCO val2017 图片上，700×700 输出的 CUDA 前处理相对 CPU 参考为：

| 指标 | 实测范围 | 说明 |
|---|---:|---|
| max absolute error | `7.15e-7–8.34e-7` | 逐元素最大差异 |
| RMSE | `6.0e-8–7.8e-8` | 1,470,000 个 FP32 元素的均方根误差 |
| relative L1 | `1.4e-8–2.5e-8` | `sum(abs(cuda-cpu))/sum(abs(cpu))` |
| cosine similarity | `1.000000000` | 输入张量方向一致 |
| CPU preprocess | `7.03–7.30 ms` | 不含 JPEG 解码 |
| CUDA preprocess | `0.325–0.350 ms` | 包含 RGB uint8 H2D 和 CUDA kernel |

以 FlashAttention FP16 engine 的单图 C++ persistent benchmark 为例，CPU 前处理为
`6.60 ms`、inference 为 `8.80 ms`、后处理为 `0.07 ms`；P1 分别为 `0.22 ms`、`8.13 ms`
和 `0.07 ms`。三段相加约从 `15.47 ms` 降至 `8.41 ms`，下降约 `45.6%`。该结果只
说明 P1 的预处理和 device-input 链路工作正常；完整 COCO mAP、连续帧吞吐和 JPEG
解码仍需单独测量，不能由这组单图结果推断。

当前实现将上传和 kernel 放在 TensorRT 相同的 persistent stream 上，并在计时边界同步，
便于准确拆分阶段。这样测得的是可复现的阶段耗时，不是 P2 异步流水的最终上限。

完整 COCO val2017（5000 张）验证结果：

| 前处理 | mAP@[.5:.95] | mAP50 | mAP75 | 平均前处理 | 平均 inference |
|---|---:|---:|---:|---:|---:|
| CPU | 0.580813 | 0.765025 | 0.630494 | 6.72 ms | 8.80 ms |
| CUDA P1 | 0.580853 | 0.765162 | 0.630586 | 0.28 ms | 8.11 ms |
| 差值 | +0.000040 | +0.000136 | +0.000092 | -6.44 ms | -0.69 ms |

P1 满足 mAP 绝对变化不超过 `0.001` 的验收门槛。FP32 engine 的 20 张子集验证也保持
mAP（`0.69731 → 0.69743`）。

当前选定的 SmoothQuant decoder engine 完整 COCO 对照同样通过：

| 前处理 | mAP@[.5:.95] | mAP50 | mAP75 | 平均前处理 | 平均 inference |
|---|---:|---:|---:|---:|---:|
| CPU | 0.580615 | 0.764990 | 0.631611 | 6.75 ms | 17.53 ms |
| CUDA P1 | 0.580423 | 0.764995 | 0.630816 | 0.29 ms | 16.78 ms |
| 差值 | -0.000192 | +0.000005 | -0.000795 | -6.46 ms | -0.75 ms |

因此 CUDA 前处理在当前量化方案和 FlashAttention FP16 方案上都满足精度门槛。

第一阶段 CUDA preprocess 的正确性门槛：

- 与 CPU 基线输出 shape、layout、dtype 完全一致；
- 归一化后的逐元素 max absolute error 目标不超过 `2e-4`；
- cosine similarity 目标不低于 `0.999999`；
- TensorRT `dets`/`labels` 不出现异常值、NaN 或类别槽位变化；
- 完整 COCO mAP 相对当前 baseline 的绝对变化不超过 `0.001`，除非报告明确说明是可接受的速度/精度折中。

性能门槛分两层：

1. P1/P2 只看 C++ steady-state，不把文件读取偶发抖动算作 CUDA kernel 性能；
2. 端到端至少需要比当前 C++ persistent 基线约 `15.62 ms` 有稳定下降，并单独报告
   预处理收益和 inference 收益。

初步目标是把前处理从约 `6.78 ms` 降到 `2 ms` 左右；这个目标需要通过实测确认，
不能把 JPEG 文件读取时间强行归入可由 resize kernel 消除的部分。

## 7. 模型侧剩余事项

已完成的模型侧事项：独立 CUTLASS plugin、plugin 嵌入 engine、LayerNorm/residual
profile 审计、QK/AV INT8 消融、CUDA Graph 和 pinned host buffer。仍需单独跟踪：

1. 为独立 plugin 扩展变长序列和其它 head 配置，并在目标端验证 TensorRT/CUDA ABI；
2. 对独立 plugin 的 12 个 attention block 补充逐层 cosine、relative L1、Softmax 熵和饱和率记录；
3. 评估 FP8 或保留 FP16/FP32 累加的更细粒度 QK/AV 低精度方案；
4. 评估多帧双缓冲 pipeline；当前单图 CUDA Graph 和 pinned host 已完成；
5. 对 CPU 后处理（当前约 `0.06–1.0 ms`）做 ROI 评估，只有目标端占比足够高时再实现 CUDA kernel。

阶段三任何结果都必须标明使用的模型 engine、plugin 版本、输入数据路径和 runtime
模式，避免把前处理收益与模型 kernel 收益混在一起。

## 8. 当前不做的事项

本阶段暂不修改模型结构、token 数、输入分辨率、训练权重或检测头逻辑；暂不把 Softmax
概率直接改成 INT8；暂不因为理论上可并行就强行 CUDA 化当前 0.07 ms 的后处理。所有
结构性改动另开实验，避免前处理 benchmark 失去可比性。
