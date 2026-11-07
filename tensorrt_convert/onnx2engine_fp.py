import sys
import numpy as np

import tensorrt as trt
from cuda.bindings import runtime as cudart

# Largest batch the engine is built to accept. Raise it if you plan to
# feed larger batches; the engine cannot exceed what the profile allows.
MAX_BATCH_SIZE = 8

def _check(result):
    # cuda.bindings calls return either an error code or a tuple whose
    # first element is the error code.
    err = result[0] if isinstance(result, tuple) else result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")

def build_engine(onnx_path: str, engine_path: str) -> bytes:
    """Build a strongly typed TensorRT engine from an ONNX file and save it."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB

    # A network with a dynamic dimension needs an optimization profile
    # giving the builder the min, optimal, and max extent of every
    # dynamic input. Without one, build_serialized_network fails.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        dims = tuple(tensor.shape)
        if -1 in dims:
            smallest = tuple(1 if d == -1 else d for d in dims)
            largest = tuple(MAX_BATCH_SIZE if d == -1 else d for d in dims)
            profile.set_shape(tensor.name, smallest, smallest, largest)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return bytes(serialized)

def run_inference(engine_bytes: bytes, input_array: np.ndarray) -> np.ndarray:
    """Deserialize the engine and run a single inference on the supplied input."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    input_name = None
    output_names = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    if input_name is None:
        raise ValueError("Engine has no input tensor!")

    context.set_input_shape(input_name, input_array.shape)
    # 动态获取输入期望的 dtype (比写死 float32 更安全)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    host_input = np.ascontiguousarray(input_array, dtype=input_dtype)

    # ==========================================
    # 3. 处理输出 (多输出)
    # ==========================================
    host_outputs = {}
    d_outputs = {}
    for name in output_names:
        out_shape = tuple(context.get_tensor_shape(name))
        # 【关键】动态获取每个输出头的 dtype，防止类型不匹配报错
        out_dtype = trt.nptype(engine.get_tensor_dtype(name))
        host_outputs[name] = np.empty(out_shape, dtype=out_dtype)

    # 4. 分配 Device 内存 & 创建 Stream
    # ==========================================
    err, d_input = cudart.cudaMalloc(host_input.nbytes); _check(err)
    for name in output_names:
        err, d_ptr = cudart.cudaMalloc(host_outputs[name].nbytes); _check(err)
        d_outputs[name] = d_ptr
    err, stream = cudart.cudaStreamCreate(); _check(err)

    # ==========================================
    # 5. H2D 拷贝 & 绑定 Tensor 地址
    # ==========================================
    _check(cudart.cudaMemcpyAsync(
        d_input, host_input.ctypes.data, host_input.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream,
    ))
    context.set_tensor_address(input_name, int(d_input))

    for name in output_names:
        context.set_tensor_address(name, int(d_outputs[name]))

    # ==========================================
    # 6. 执行推理
    # ==========================================
    context.execute_async_v3(stream)

    # ==========================================
    # 7. D2H 拷贝 (多输出)
    # ==========================================
    for name in output_names:
        host_out = host_outputs[name]
        _check(cudart.cudaMemcpyAsync(
            host_out.ctypes.data, d_outputs[name], host_out.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream,
        ))

    _check(cudart.cudaStreamSynchronize(stream))

    # ==========================================
    # 8. 清理内存
    # ==========================================
    cudart.cudaFree(d_input)
    for d_ptr in d_outputs.values():
        cudart.cudaFree(d_ptr)
    cudart.cudaStreamDestroy(stream)

    # 返回一个字典：{ "output_name_1": np.array, "output_name_2": np.array, ... }
    return host_outputs


if __name__ == "__main__":
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "resnet50-v1-12.onnx"
    engine_path = "model.engine"
    engine_bytes = build_engine(onnx_path, engine_path)

    # Replace this with a preprocessed image batch in your own application.

    import cv2
    demo_image = cv2.imread("/home/rton/rfdetr/img/000000085772.jpg")
    dummy_input = demo_image.astype(np.float32)
    outputs_dict = run_inference(engine_bytes, dummy_input)

    # 处理多输出结果
    print("--- Inference Results ---")
    for out_name, out_array in outputs_dict.items():
        print(f"Output name: '{out_name}'")
        print(f"  -> Shape: {out_array.shape}")
        print(f"  -> Dtype: {out_array.dtype}")

        # 示例：如果某个输出是分类 logits (比如 shape 为 [1, 1000])
        if out_array.ndim == 2 and out_array.shape[1] > 10:
            print(f"  -> Top-1 class index: {int(np.argmax(out_array[0]))}")
