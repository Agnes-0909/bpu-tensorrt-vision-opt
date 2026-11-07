cmake_minimum_required(VERSION 3.18)
project(rfdetr_tensorrt_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "$ENV{OpenCV_DIR}" CACHE PATH "OpenCV package configuration directory")
if(NOT OpenCV_DIR)
    set(OpenCV_DIR "/home/rton/opencv-4.9.0-install/lib/cmake/opencv4")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT installation root")
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "/home/rton/TensorRT/TensorRT-11.4.0.106")
endif()
set(CUDA_ROOT "$ENV{CUDA_HOME}" CACHE PATH "CUDA installation root")
if(NOT CUDA_ROOT)
    set(CUDA_ROOT "/usr/local/cuda")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS "${TENSORRT_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
find_library(NVINFER_LIBRARY nvinfer PATHS "${TENSORRT_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin PATHS "${TENSORRT_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
find_library(CUDART_LIBRARY cudart PATHS "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/targets/x86_64-linux/lib" REQUIRED)

add_executable(rfdetr_trt
    src/infer.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/main.cpp
)
add_executable(rfdetr_benchmark
    src/infer.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/benchmark.cpp
)
target_include_directories(rfdetr_trt PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CUDA_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_include_directories(rfdetr_benchmark PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CUDA_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rfdetr_trt PRIVATE "${NVINFER_LIBRARY}" "${NVINFER_PLUGIN_LIBRARY}" "${CUDART_LIBRARY}" ${OpenCV_LIBS} dl)
target_link_libraries(rfdetr_benchmark PRIVATE "${NVINFER_LIBRARY}" "${NVINFER_PLUGIN_LIBRARY}" "${CUDART_LIBRARY}" ${OpenCV_LIBS} dl)
target_compile_options(rfdetr_trt PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
target_compile_options(rfdetr_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
if(OpenCV_INSTALL_PATH)
    set(OPENCV_LIB_DIR "${OpenCV_INSTALL_PATH}/lib")
else()
    get_filename_component(OPENCV_LIB_DIR "${OpenCV_DIR}/../.." ABSOLUTE)
endif()
set_target_properties(rfdetr_trt PROPERTIES BUILD_RPATH "${TENSORRT_ROOT}/lib;${CUDA_ROOT}/lib64;${CUDA_ROOT}/targets/x86_64-linux/lib;${OPENCV_LIB_DIR}")
set_target_properties(rfdetr_benchmark PROPERTIES BUILD_RPATH "${TENSORRT_ROOT}/lib;${CUDA_ROOT}/lib64;${CUDA_ROOT}/targets/x86_64-linux/lib;${OPENCV_LIB_DIR}")
