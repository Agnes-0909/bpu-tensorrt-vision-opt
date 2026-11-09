# RF-DETR fused attention / FlashAttention plugin 原型

这个目录是阶段二的自定义算子实验。它把 encoder 中的

```text
QK^T -> Softmax -> AV
```

三个 TensorRT 层替换成一个 TensorRT plugin。原生 CUDA 版本名为
`RFDetrFusedAttention`，定义 `TORCH_FLASH` 编译的版本名为
`RFDetrTorchFlashAttention`。当前固定形状为 `B=1、H=12、S=1226、D=64`。

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
性能验证原型，还不是可独立分发的 TensorRT engine。后续生产化需要去除 libtorch
依赖，直接接入兼容的 FlashAttention/CUTLASS kernel，并逐层验证 12 个 attention
输出后再进行 COCO 全量评估。
