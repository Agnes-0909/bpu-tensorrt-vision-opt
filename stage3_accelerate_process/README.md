# Stage 3：前处理与后处理 CUDA 加速

阶段三承接 `stage2_advanced_opt`。阶段二已经完成 attention 热点分析、FlashAttention
原型和 persistent TensorRT runtime；本阶段已经完成 P1 前处理原型：OpenCV 负责图像
解码，CUDA 负责 RGB uint8 上传、双线性 resize、ImageNet normalize 和 HWC→NCHW，结果
直接写入 TensorRT 的 persistent input device buffer。

阶段二留下的模型侧事项也在本阶段完成了验证，状态如下：

| 待办 | 当前状态 | 后续动作 |
|---|---|---|
| FlashAttention 的 libtorch 依赖 | PyTorch SDPA 版本仍保留作性能上限对照 | 生产候选改用独立 CUTLASS plugin |
| engine 依赖额外 plugin `.so` | 动态 engine 仍需注册 plugin creator；CUTLASS engine 已验证可用 `--setPluginsToSerialize` 嵌入 plugin | 部署优先使用嵌入后的 engine |
| 独立 CUDA/CUTLASS plugin | 已实现 `RFDetrCutlassFlashAttention`，不链接 libtorch；完整 COCO mAP `0.5806461` | 当前固定 `B=1,H=12,S=1226,D=64`，变长 shape 另行扩展 |
| LayerNorm/residual 融合 | 已完成 profile 审计；没有独立 LayerNorm，TensorRT 已融合主要 pointwise/residual 路径 | 暂不手工改图，避免引入额外 reformat |
| QK/AV 选择性低精度 | QK-only、AV-only、QK+AV INT8 均完成 COCO 子集验证，精度损失超门槛，全部拒绝 | 保持 QK/Softmax/AV 高精度 |

阶段三优先优化前处理，但以上模型侧任务仍是最终生产方案的必要条件。

| 文件 | 作用 |
|---|---|
| [`STAGE3_PREPROCESS_OPTIMIZATION_PLAN.md`](STAGE3_PREPROCESS_OPTIMIZATION_PLAN.md) | 基线、正确性约束、CUDA 前处理分层方案、实验矩阵、验收门槛和模型侧待办 |
| `cuda_preprocess.cu/.h` | P1 的 resize、normalize、HWC→NCHW CUDA fused kernel，以及临时 RGB device buffer |
| `cuda_postprocess.cu/.h` | 暂未实现；CPU 后处理约 `0.06–1.0 ms`，当前 ROI 很低，保留为后续可选项 |
| `benchmark_preprocess.cpp` | CPU/CUDA 前处理数值误差和独立耗时对比 |
| `../tensorrt_infer/src/benchmark.cpp` | `per_call`、`persistent`、`cuda_preprocess`、`cuda_preprocess_pinned`、`cuda_preprocess_async` 五种端到端分段 benchmark |
| `../tensorrt_infer/src/infer.cpp` | persistent stream/device buffer、device-input API 和 CUDA Graph capture/replay |
| `evaluate_coco_cuda_preprocess.cpp` | 使用 CPU 或 CUDA 前处理生成 COCO prediction JSON |
| `evaluate_coco_preprocess.py` | 调用 pycocotools 计算 mAP，并比较多组 prediction JSON |
| `benchmark_nvjpeg_preprocess.cpp` | nvJPEG GPU 解码到 RGBI 后接 CUDA fused preprocess 的单图 benchmark |
| `experiments/` | 后续本地 engine、日志、预测和 profile 产物；不作为源码提交 |

在当前 RTX 5060 Ti、700×700、单图 steady-state 测试中，P1 前处理（包含 pageable
host 上的 RGB uint8 H2D，不包含 JPEG 文件解码）约为 `0.21–0.35 ms`，CPU 前处理约为
`7.0–7.3 ms`，输入张量的最大绝对误差小于 `1e-6`、RMSE 约 `1e-7`、cosine 为
`1.0`。以 FlashAttention FP16 engine 为例，C++ persistent 基线为前处理 `6.60 ms`
和 inference `8.80 ms`，P1 为前处理 `0.22 ms` 和 inference `8.13 ms`；三段相加的
steady-state 端到端时间从约 `15.47 ms` 降到约 `8.41 ms`，约下降 `45.6%`。这是单图
分段 benchmark，尚未替代完整 COCO 评测，也不包含 JPEG 解码和 I/O。

完整 COCO val2017 验证已经完成。FlashAttention FP16 engine 使用完全相同的模型和
后处理，仅切换 CPU/CUDA 前处理：

| 前处理 | mAP@[.5:.95] | mAP50 | mAP75 | 平均前处理 | 平均 inference |
|---|---:|---:|---:|---:|---:|
| CPU | 0.580813 | 0.765025 | 0.630494 | 6.72 ms | 8.80 ms |
| CUDA P1 | 0.580853 | 0.765162 | 0.630586 | 0.28 ms | 8.11 ms |
| 差值 | +0.000040 | +0.000136 | +0.000092 | -6.44 ms | -0.69 ms |

P1 通过了当前 `0.001` 的 mAP 绝对变化门槛。FP16 FlashAttention 对单个输入的微小
差异比 FP32 更敏感，但 5000 张完整评测没有产生可见精度损失。

P2 的 pinned host buffer 已完成第一版：在前处理 benchmark 中，pageable RGB 上传约
`0.394 ms`，pinned RGB 上传约 `0.246 ms`，输入误差仍为 `8.34e-7`。FlashAttention
端到端 `cuda_preprocess_pinned` 测试中的前处理约 `0.151 ms`，inference 约 `8.10 ms`。
该测试假设图像解码结果已经写入 pinned buffer；当前示例中的一次性 pageable→pinned
复制不计入 steady-state，因此连续帧场景需要让解码器直接写入 pinned memory 才能获得
完整收益。

CUDA Graph 也已完成固定 shape 的验证。FlashAttention engine 的推理段约从 `8.10 ms`
降到 `7.21 ms`；完整 COCO 5000 张 Graph mAP 为 `0.580853`，与 CUDA P1 路径一致，
相对 CPU 前处理基线差值为 `+0.000040`。

P3 nvJPEG 原型也已跑通：640×428 JPEG 解码到 GPU RGBI，再接 fused preprocess，平均
`0.747 ms`，输入 max absolute error `8.34e-7`。这个数字包含 nvJPEG 解码和 CUDA
resize/normalize/layout，但不包含文件读取；它还没有接入完整 TensorRT COCO evaluator，
需要在目标端确认 nvJPEG backend、文件 I/O 和连续帧吞吐后才能替代 OpenCV。

当前选定的 SmoothQuant decoder engine 也完成了完整 COCO 对照：

| 前处理 | mAP@[.5:.95] | mAP50 | mAP75 | 平均前处理 | 平均 inference |
|---|---:|---:|---:|---:|---:|
| CPU | 0.580615 | 0.764990 | 0.631611 | 6.75 ms | 17.53 ms |
| CUDA P1 | 0.580423 | 0.764995 | 0.630816 | 0.29 ms | 16.78 ms |
| 差值 | -0.000192 | +0.000005 | -0.000795 | -6.46 ms | -0.75 ms |

这说明 P1 的收益和精度保持不依赖 FlashAttention plugin；在当前量化方案上同样成立。

运行方式：

```bash
conda activate TensorRT
cd /home/rton/rfdetr/tensorrt_infer
CUDA_ARCHITECTURES=120 ./build.sh

# 只测 CPU/CUDA 前处理的数值和耗时
./build/rfdetr_preprocess_benchmark \
  /home/rton/val2017/val2017/000000082085.jpg 700 700 60 20

# 端到端分段计时：最后一个参数选择 persistent 或 cuda_preprocess
export LD_LIBRARY_PATH=/home/rton/miniconda3/envs/torch/lib/python3.12/site-packages/torch/lib:/home/rton/TensorRT/TensorRT-11.4.0.106/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/home/rton/rfdetr/stage2_advanced_opt/experiments/fused_attention_plugin/build/libtorch_sdpa_attention.so
./build/rfdetr_benchmark \
  /home/rton/stage2_advanced_opt/experiments/fused_attention_plugin/rfdetr-xlarge-torch-sdpa-half.engine \
  /home/rton/val2017/val2017/000000082085.jpg 100 20 0.5 -1 cuda_preprocess

# 异步串接：不在 CUDA preprocess 和 TensorRT enqueue 之间插入主机同步
./build/rfdetr_benchmark \
  /home/rton/stage2_advanced_opt/experiments/fused_attention_plugin/rfdetr-xlarge-torch-sdpa-half.engine \
  /home/rton/val2017/val2017/000000082085.jpg 100 20 0.5 -1 cuda_preprocess_async

# pinned host RGB 上传（假设解码结果已在 pinned buffer）
./build/rfdetr_benchmark \
  /home/rton/stage2_advanced_opt/experiments/fused_attention_plugin/rfdetr-xlarge-torch-sdpa-half.engine \
  /home/rton/val2017/val2017/000000082085.jpg 100 20 0.5 -1 cuda_preprocess_pinned

# CUDA Graph：固定输入 shape 下重放 TensorRT enqueue
./build/rfdetr_benchmark \
  /home/rton/stage2_advanced_opt/experiments/fused_attention_plugin/rfdetr-xlarge-torch-sdpa-half.engine \
  /home/rton/val2017/val2017/000000082085.jpg 100 20 0.5 -1 cuda_graph

# nvJPEG GPU 解码 + CUDA 前处理
./build/rfdetr_nvjpeg_preprocess_benchmark \
  /home/rton/val2017/val2017/000000082085.jpg 700 700 60 20

# 生成 CUDA 前处理的 COCO 预测并计算 mAP
./build/rfdetr_coco_cuda_preprocess \
  /home/rton/stage2_advanced_opt/experiments/fused_attention_plugin/rfdetr-xlarge-torch-sdpa-half.engine \
  /home/rton/val2017/val2017 /home/rton/rfdetr/stage3_accelerate_process/experiments/coco_cuda.json 0 0 cuda
python /home/rton/rfdetr/stage3_accelerate_process/evaluate_coco_preprocess.py \
  --predictions /home/rton/rfdetr/stage3_accelerate_process/experiments/coco_cuda.json
```

PyTorch SDPA FlashAttention engine 仍需先加载阶段二的 plugin `.so` 和 libtorch 运行库；
生产候选应使用不依赖 libtorch 的 CUTLASS plugin，并优先使用已经嵌入 plugin 的 engine。
普通 FP32 或 SmoothQuant engine 不需要这一步。`cuda_preprocess` 为了清楚分离阶段，在 TensorRT
enqueue 前同步同一条 persistent stream；`cuda_preprocess_async` 已验证同一 stream
直接串接，FlashAttention engine 单图端到端约 `8.23 ms`，相比分离同步模式约 `8.35 ms`
略有下降。pinned host buffer、双缓冲、GPU JPEG/NVDEC 解码和 CUDA 后处理仍是后续阶段。

完整方案见 [`STAGE3_PREPROCESS_OPTIMIZATION_PLAN.md`](STAGE3_PREPROCESS_OPTIMIZATION_PLAN.md)。
