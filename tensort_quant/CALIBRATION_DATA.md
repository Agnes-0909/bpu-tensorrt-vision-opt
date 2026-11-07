# XLarge 校准数据集

本次最终 XLarge INT8 方案使用 COCO val2017 作为代表性校准集，原始图片目录为：

`/home/rton/val2017/val2017`

从目录中的 JPEG 文件按文件名排序后，用 `numpy.linspace` 均匀采样 64 张图片。完整清单和 manifest 在本地生成到 `artifacts_xlarge_conv/`，其中包含机器相关的绝对路径，因此不进入 Git。

校准预处理与评测一致：BGR 转 RGB、torchvision 风格 bilinear resize 到 700x700（antialias=False），再使用 ImageNet mean/std 归一化。

重新生成量化模型时可以指定其他目录和数量：

```bash
/home/rton/miniconda3/envs/torch/bin/python quantize_int8.py \
  --model ../python_infer/output/rfdetr-xlarge.onnx \
  --images /path/to/calibration/images \
  --calibration-count 64 \
  --quantize-ops Conv
```
