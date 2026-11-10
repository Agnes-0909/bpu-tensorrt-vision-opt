# RF-DETR fused attention / FlashAttention plugin 原型

这个目录是阶段二的自定义算子实验。它把 encoder 中的

```text
QK^T -> Softmax -> AV
```

三个 TensorRT 层替换成一个 TensorRT plugin。原生 CUDA 版本名为
`RFDetrFusedAttention`，定义 `TORCH_FLASH` 编译的版本名为
`RFDetrTorchFlashAttention`，CUTLASS 版本名为 `RFDetrCutlassFlashAttention`。
当前固定形状为 `B=1、H=12、S=1226、D=64`。

图改写脚本保留原图的 Q/V `[B,H,S,D]` 和 K `[B,H,D,S]` 输入。使用
`--torch-flash` 时会在 plugin 前显式转置 K，FlashAttention 实际接收的
Q/K/V 都是 `[B,H,S,D]`。plugin 内部完成 QK^T、稳定 Softmax 和 AV，不落地
`[B,H,S,S]` score/probability 张量。

## 原生 tiled CUDA 版本

```bash
cd /home/rton/rfdetr/stage2_advanced_opt/experiments/fused_attention_plugin
/usr/local/cuda/bin/nvcc -std=c++17 -O3 --use_fast_math \
  -gencode arch=compute_120,code=sm_120 -Xcompiler -fPIC -shared \
  fused_attention_plugin.cu -o build/libfused_attention.so \
  -I/home/rton/TensorRT/TensorRT-11.4.0.106/include \
  -L/home/rton/TensorRT/TensorRT-11.4.0.106/lib -lnvinfer -lcudart

conda run -n TensorRT python rewrite_to_plugin.py \
  --input ../../../tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx \
  --output rfdetr-xlarge-plugin.onnx

/home/rton/TensorRT/TensorRT-11.4.0.106/bin/trtexec \
  --onnx=rfdetr-xlarge-plugin.onnx --saveEngine=rfdetr-xlarge-plugin.engine \
  --dynamicPlugins=build/libfused_attention.so --skipInference \
  --builderOptimizationLevel=5
```

## PyTorch FlashAttention 版本

TensorRT 环境没有 PyTorch runtime，因此该实验从 `torch` conda 环境链接
`libtorch_cuda.so`，通过 plugin 调用 public
`at::scaled_dot_product_attention`。它使用图的 `[B,H,S,D]` 布局，由 PyTorch
CUDA dispatcher 选择 FlashAttention backend。不能直接调用底层
`_flash_attention_forward_no_dropout_inplace`，后者使用另一种 packed layout，
会导致第一层 attention 数值错误。

```bash
./build_torch_sdpa_plugin.sh

# 路径不同的时候覆盖默认环境变量
TENSORRT_ROOT=/path/to/TensorRT \
TORCH_ROOT=/path/to/conda/envs/torch \
CUDA_HOME=/usr/local/cuda \
./build_torch_sdpa_plugin.sh

conda run -n TensorRT python rewrite_to_plugin.py \
  --input ../../../tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx \
  --output rfdetr-xlarge-torch-sdpa-half.onnx --half --torch-flash
```

`--half` 只在 12 个 attention 子图的 Q/K/V 输入和输出边界插入 FP16 Cast，
其余 projection、残差、LayerNorm、FFN、decoder 和 head 的图范围不变。
运行时需要提供：

```bash
export LD_LIBRARY_PATH=/home/rton/miniconda3/envs/torch/lib/python3.12/site-packages/torch/lib:/home/rton/TensorRT/TensorRT-11.4.0.106/lib:$LD_LIBRARY_PATH
/home/rton/TensorRT/TensorRT-11.4.0.106/bin/trtexec \
  --onnx=rfdetr-xlarge-torch-sdpa-half.onnx \
  --saveEngine=rfdetr-xlarge-torch-sdpa-half.engine \
  --dynamicPlugins=build/libtorch_sdpa_attention.so \
  --skipInference --builderOptimizationLevel=5
```

engine 构建完成后，推理阶段不再需要 ONNX 或 `rewrite_to_plugin.py`；但在
TensorRT 反序列化 engine 之前必须注册 `RFDetrTorchFlashAttention` creator。
对当前 C++ 示例程序可以直接这样运行：

```bash
export LD_LIBRARY_PATH=/home/rton/miniconda3/envs/torch/lib/python3.12/site-packages/torch/lib:/home/rton/TensorRT/TensorRT-11.4.0.106/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$PWD/build/libtorch_sdpa_attention.so:$LD_PRELOAD
/home/rton/rfdetr/tensorrt_infer/build/rfdetr_benchmark \
  rfdetr-xlarge-torch-sdpa-half.engine \
  /home/rton/rfdetr/img/000000085772.jpg 500 100 0.5 -1 persistent
```

`trtexec` 则使用 `--dynamicPlugins=build/libtorch_sdpa_attention.so`。如果完全不
加载这个 `.so`，engine 反序列化会失败并报：
`Cannot find plugin: RFDetrTorchFlashAttention`。这是因为 engine 序列化了 plugin
节点的类型、版本和参数，但当前构建没有把 plugin 实现及其 libtorch 依赖打包成
独立 engine。

## 当前结果

- tiled CUDA plugin 第一层输出 cosine 为 `0.9999999`，但全图约 `108 ms`，明显慢于 baseline 的 `16.7 ms`。
- PyTorch SDPA/FlashAttention plugin 在同一 RTX 5060 Ti、`--noDataTransfers --noCudaGraph` 条件下约 `8.03 ms`（50 次，P95 `8.51 ms`），相对 baseline 约降低 `52%`。
- 修正 K 的 layout 后，第一层 FlashAttention 输出 cosine 为 `0.99999947`、RMSE `2.07e-4`。同一 Polygraphy 输入下，完整输出 `labels` cosine 为 `0.998630`，`dets` cosine 为 `0.773431`。这证明 kernel 路径有性能潜力，但当前 FP16 attention 仍需 COCO 精度验证，不能直接作为最终模型方案。
- COCO val2017 前 100 张图片上，mAP@[.5:.95] 为 `0.635750`，baseline 为 `0.638222`，下降 `0.247` 个百分点；persistent C++ inference 均值为 `9.595 ms`，baseline 为 `18.684 ms`，降低约 `48.6%`。因此它是有明显收益但尚未通过当前精度门限的候选。
- 完整 COCO val2017 5000 张图片上，mAP@[.5:.95] 为 `0.5803911`，同一阶段 baseline 为 `0.5802505`，差值 `+0.0001406`。完整评估的 persistent inference 均值为 `9.654 ms`，预处理 `30.525 ms`，后处理 `0.983 ms`，三者相加的端到端均值约 `41.162 ms`。

Polygraphy 输出可复核：

```bash
conda run -n TensorRT python compare_polygraphy_outputs.py \
  baseline_outputs_trt.json torch_sdpa_outputs.json
```

在同一台机器上做 COCO 前 100 张评估时，需要让 TensorRT Python runtime 先加载
plugin：

```bash
export LD_PRELOAD=$PWD/build/libtorch_sdpa_attention.so:$LD_PRELOAD
/home/rton/miniconda3/envs/torch/bin/python ../../../evaluate_coco_tensorrt.py \
  --engine rfdetr-xlarge-torch-sdpa-half.engine --limit 100 \
  --output coco_sdpa_predictions.json --report coco_sdpa_100.json
```

`getCreators` 和 `setLoggerFinder` 是 TensorRT `--dynamicPlugins` loader 要求的
导出符号。编译目标必须包含 `sm_120`，否则 RTX 5060 Ti 运行时可能出现
`provided PTX was compiled with an unsupported toolchain`。`libtorch_sdpa_attention.so`
及其 PyTorch/CUDA 依赖必须随运行时提供，所以该版本是 FlashAttention kernel
性能验证原型。下面的 CUTLASS 版本用于去除这条 libtorch 依赖。

两条依赖链可以用下面的脚本直接检查；它会确认参考 plugin 确实链接 torch，而独立
CUTLASS plugin 不包含 `libtorch`/`libc10`：

```bash
./check_plugin_dependencies.sh
```

## 独立 CUTLASS FlashAttention plugin

`cutlass_flash_attention_plugin.cu` 使用 CUTLASS 41 号 fused multi-head attention
的 online Softmax/Tensor Core kernel，plugin 自己负责把 RF-DETR 的 Q/V
`[B,H,S,D]` 和 K `[B,H,D,S]` 临时重排为 CUTLASS 所需的 BMHK 布局，再把输出重排回来。
它只链接 TensorRT 和 CUDA，不包含 `libtorch.so`、`libtorch_cuda.so` 或 `libc10`。
`cutlass_kernel_forward.h` 是从 CUTLASS 的 `kernel_forward.h` 提取并改名内部辅助符号后的
固定版本；该头文件对应 CUTLASS commit
`b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd`。完整 CUTLASS 源码作为构建依赖放在仓库外，
避免把整个第三方仓库提交进项目。

```bash
cd /home/rton
git clone --depth 1 https://github.com/NVIDIA/cutlass.git cutlass
cd /home/rton/rfdetr/stage2_advanced_opt/experiments/fused_attention_plugin
CUTLASS_ROOT=/home/rton/cutlass ./build_cutlass_flash_plugin.sh

conda run -n TensorRT python rewrite_to_plugin.py \
  --input ../../../tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx \
  --output rfdetr-xlarge-cutlass-flash-half.onnx --half --cutlass-flash

# 构建阶段仍要加载 .so，让 ONNX parser 找到 creator；同时把 plugin 库嵌入 engine。
/home/rton/TensorRT/TensorRT-11.4.0.106/bin/trtexec \
  --onnx=rfdetr-xlarge-cutlass-flash-half.onnx \
  --saveEngine=rfdetr-xlarge-cutlass-flash-half-embedded.engine \
  --dynamicPlugins=build/libcutlass_flash_attention.so \
  --setPluginsToSerialize=build/libcutlass_flash_attention.so \
  --skipInference --builderOptimizationLevel=5
```

嵌入后的 engine 已验证可以在不设置 `LD_PRELOAD`、不传
`--dynamicPlugins` 的情况下反序列化和推理；普通动态 plugin engine 仍需要在
反序列化前注册 creator。嵌入 plugin 只解决运行时的外部 `.so` 查找，仍要求部署端的
TensorRT/CUDA ABI、GPU 架构和 engine 构建环境匹配。

本机 RTX 5060 Ti、TensorRT 11.4、`B=1,H=12,S=1226,D=64` 的结果：

| 方案 | libtorch 依赖 | engine 纯 GPU latency | COCO val2017 mAP@[.5:.95] |
|---|---|---:|---:|
| 原生 tiled CUDA plugin | 无 | `108.4 ms` | 未作为候选 |
| PyTorch SDPA/FlashAttention | 有 | `8.03 ms` | `0.5803911` |
| 独立 CUTLASS plugin（动态加载） | 无 | `9.99 ms` | `0.5806461` |
| 独立 CUTLASS plugin（plugin 嵌入 engine） | 无 | `9.86 ms`（10 次复测） | 与上行相同 |

独立 CUTLASS 版本相对 SmoothQuant baseline（mAP `0.5802505`）没有精度下降，反而有
`+0.000396` 的测量差异；它比 libtorch 版本慢约 `1.8 ms`，但已经满足“无 PyTorch
运行时”的部署条件。当前 plugin 仍针对固定 encoder 形状，动态序列长度和其它 head
配置需要单独实例化模板并重新验证。
