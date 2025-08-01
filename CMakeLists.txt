﻿cmake_minimum_required(VERSION 3.12)
project(yolov11-tensorrt)

find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DAPI_EXPORTS)

# Suppress OpenCV virtual function override warnings in CUDA compilation
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -diag-suppress 611)

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src/)

# Add source files
set(SOURCES
    main.cpp
    src/YOLOv11.cpp
)

# Add CUDA source files
set(CUDA_SOURCES src/preprocess.cu src/gpu_nms.cu)

# Add headers
set(HEADERS
    src/YOLOv11.h
    src/macros.h
    src/logging.h
    src/cuda_utils.h
    src/preprocess.h
    src/common.h
    src/gpu_nms.h
)

# Find and include OpenCV
set(OpenCV_DIR "your OpenCV build directory path")

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "your tensorrt path")

# Include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)

# Create an executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)