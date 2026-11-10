cmake_minimum_required(VERSION 3.18)
project(rfdetr_tensorrt_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures for preprocessing kernels")
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
find_library(NVJPEG_LIBRARY nvjpeg PATHS "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/targets/x86_64-linux/lib" REQUIRED)

add_library(rfdetr_cuda_preprocess OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/../stage3_accelerate_process/cuda_preprocess.cu"
)
target_include_directories(rfdetr_cuda_preprocess PRIVATE "${CUDA_ROOT}/include")
set_target_properties(rfdetr_cuda_preprocess PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(rfdetr_trt
    src/infer.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/main.cpp
    $<TARGET_OBJECTS:rfdetr_cuda_preprocess>
)
add_executable(rfdetr_benchmark
    src/infer.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/benchmark.cpp
    $<TARGET_OBJECTS:rfdetr_cuda_preprocess>
)
add_executable(rfdetr_preprocess_benchmark
    "${CMAKE_CURRENT_SOURCE_DIR}/../stage3_accelerate_process/benchmark_preprocess.cpp"
    src/preprocess.cpp
    $<TARGET_OBJECTS:rfdetr_cuda_preprocess>
)
add_executable(rfdetr_coco_cuda_preprocess
    "${CMAKE_CURRENT_SOURCE_DIR}/../stage3_accelerate_process/evaluate_coco_cuda_preprocess.cpp"
    src/infer.cpp
    src/preprocess.cpp
    $<TARGET_OBJECTS:rfdetr_cuda_preprocess>
)
add_executable(rfdetr_nvjpeg_preprocess_benchmark
    "${CMAKE_CURRENT_SOURCE_DIR}/../stage3_accelerate_process/benchmark_nvjpeg_preprocess.cpp"
    src/preprocess.cpp
    $<TARGET_OBJECTS:rfdetr_cuda_preprocess>
)
target_include_directories(rfdetr_trt PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CUDA_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../stage3_accelerate_process" ${OpenCV_INCLUDE_DIRS})
target_include_directories(rfdetr_benchmark PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CUDA_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../stage3_accelerate_process" ${OpenCV_INCLUDE_DIRS})
target_include_directories(rfdetr_preprocess_benchmark PRIVATE "${CUDA_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/../stage3_accelerate_process" ${OpenCV_INCLUDE_DIRS})
target_include_directories(rfdetr_coco_cuda_preprocess PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CUDA_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/../stage3_accelerate_process" ${OpenCV_INCLUDE_DIRS})
target_include_directories(rfdetr_nvjpeg_preprocess_benchmark PRIVATE "${CUDA_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/../stage3_accelerate_process" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rfdetr_trt PRIVATE "${NVINFER_LIBRARY}" "${NVINFER_PLUGIN_LIBRARY}" "${CUDART_LIBRARY}" ${OpenCV_LIBS} dl)
target_link_libraries(rfdetr_benchmark PRIVATE "${NVINFER_LIBRARY}" "${NVINFER_PLUGIN_LIBRARY}" "${CUDART_LIBRARY}" ${OpenCV_LIBS} dl)
target_link_libraries(rfdetr_preprocess_benchmark PRIVATE "${CUDART_LIBRARY}" ${OpenCV_LIBS})
target_link_libraries(rfdetr_coco_cuda_preprocess PRIVATE "${NVINFER_LIBRARY}" "${NVINFER_PLUGIN_LIBRARY}" "${CUDART_LIBRARY}" ${OpenCV_LIBS} dl)
target_link_libraries(rfdetr_nvjpeg_preprocess_benchmark PRIVATE "${NVJPEG_LIBRARY}" "${CUDART_LIBRARY}" ${OpenCV_LIBS})
target_compile_options(rfdetr_trt PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
target_compile_options(rfdetr_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
target_compile_options(rfdetr_preprocess_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
target_compile_options(rfdetr_coco_cuda_preprocess PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
target_compile_options(rfdetr_nvjpeg_preprocess_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
if(OpenCV_INSTALL_PATH)
    set(OPENCV_LIB_DIR "${OpenCV_INSTALL_PATH}/lib")
else()
    get_filename_component(OPENCV_LIB_DIR "${OpenCV_DIR}/../.." ABSOLUTE)
endif()
set_target_properties(rfdetr_trt PROPERTIES BUILD_RPATH "${TENSORRT_ROOT}/lib;${CUDA_ROOT}/lib64;${CUDA_ROOT}/targets/x86_64-linux/lib;${OPENCV_LIB_DIR}")
set_target_properties(rfdetr_benchmark PROPERTIES BUILD_RPATH "${TENSORRT_ROOT}/lib;${CUDA_ROOT}/lib64;${CUDA_ROOT}/targets/x86_64-linux/lib;${OPENCV_LIB_DIR}")
set_target_properties(rfdetr_preprocess_benchmark PROPERTIES BUILD_RPATH "${CUDA_ROOT}/lib64;${CUDA_ROOT}/targets/x86_64-linux/lib;${OPENCV_LIB_DIR}")
set_target_properties(rfdetr_coco_cuda_preprocess PROPERTIES BUILD_RPATH "${TENSORRT_ROOT}/lib;${CUDA_ROOT}/lib64;${CUDA_ROOT}/targets/x86_64-linux/lib;${OPENCV_LIB_DIR}")
set_target_properties(rfdetr_nvjpeg_preprocess_benchmark PROPERTIES BUILD_RPATH "${CUDA_ROOT}/lib64;${CUDA_ROOT}/targets/x86_64-linux/lib;${OPENCV_LIB_DIR}")
