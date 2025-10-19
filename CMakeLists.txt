cmake_minimum_required(VERSION 3.18)
project(YoloTensorRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT paths (adjust these paths according to your TensorRT installation)
set(TENSORRT_ROOT /usr/local/TensorRT-10.12.0.19)
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)

# Include directories
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Link directories
link_directories(${TENSORRT_LIB_DIR})

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

# Source files
set(SOURCES
    yolov5_d.cpp
    yolov5_d11.cpp
    yololayer.cpp
    yololayer.cu
)

# Create shared library for the plugin
add_library(yolo_plugin SHARED ${SOURCES})

# Link libraries
target_link_libraries(yolo_plugin
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
    cublas
    curand
    cudnn
)

# Set properties
set_target_properties(yolo_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create executable
add_executable(yolo_test test.cpp)
target_link_libraries(yolo_test yolo_plugin)

# Install targets
install(TARGETS yolo_plugin DESTINATION lib)
install(FILES yololayer.h DESTINATION include)