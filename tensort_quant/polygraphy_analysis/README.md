# Polygraphy 共用输入

`.inputs.json` 是 Polygraphy/ONNX Runtime/TensorRT 对比使用的固定输入文件。它保证 FP32、INT8 和不同量化算法在同一输入上比较中间张量。

正式 FP32 ONNX 不在这里重复保存，统一使用：

```text
/home/rton/rfdetr/python_infer/output/rfdetr-xlarge.onnx
```

这样可以避免保留一份 447 MiB 的重复模型副本。
