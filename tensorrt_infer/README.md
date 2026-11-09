# TensorRT inference

This executable loads the RF-DETR TensorRT engine, applies the same preprocessing
as `python_infer/onnx_infer.py`, and prints decoded detections.

## Build

The default paths target the TensorRT and CUDA installations used for this
project. Override them when needed:

```bash
TENSORRT_ROOT=/path/to/TensorRT CUDA_HOME=/usr/local/cuda ./build.sh
```

The build requires TensorRT 11, CUDA, OpenCV 4.x (`core`, `imgproc`, and
`imgcodecs`), and a C++17 compiler. By default CMake uses the standalone OpenCV
installation at `/home/rton/opencv-4.9.0-install`; override it with
`OpenCV_DIR` when needed.

## Run

From the repository root (the executable also searches parent directories for
the default files when launched from `tensorrt_infer/build`):

```bash
./tensorrt_infer/build/rfdetr_trt
```

Or provide explicit paths and a score threshold:

```bash
./tensorrt_infer/build/rfdetr_trt model.engine img/000000085772.jpg 0.5 -1
```

Run the 5000-iteration baseline benchmark (100 warmup iterations):

```bash
./tensorrt_infer/build/rfdetr_benchmark model.engine img/000000085772.jpg 5000 100
```

The optional seventh argument selects the runtime resource policy. `per_call`
keeps the original reference path, which creates/destroys a stream and allocates
device buffers for every inference. `persistent` allocates them on the first
call and reuses them:

```bash
./tensorrt_infer/build/rfdetr_benchmark model.engine img/000000085772.jpg 300 100 0.5 -1 persistent
```

The benchmark reads the image once before warmup. Its preprocessing timing is
resize plus normalization; inference includes H2D/D2H copies and synchronization.
The `per_call` mode also includes CUDA allocations, while `persistent` reuses the
stream and device buffers. Postprocessing includes sigmoid decoding and sorting.

The fourth argument selects the background class slot. `-1` follows the Python
reference and removes the last exported slot; `0` removes slot zero.
