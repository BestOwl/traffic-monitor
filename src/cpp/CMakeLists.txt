cmake_minimum_required(VERSION 3.10)
project(SkiingRobot VERSION 0.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT WIN32)
    link_libraries(stdc++fs)
endif()

# OpenCV
if (NOT OpenCV_DIR)
    find_path(OpenCV_DIR OpenCVConfig.cmake HINTS $ENV{OpenCV_DIR}/lib)
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA Toolkit
find_package(CUDA REQUIRED)
message(STATUS "Found CUDA_TOOLKIT_ROOT_DIR at ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "Found CUDA headers at ${CUDA_INCLUDE_DIR}")
MESSAGE(STATUS "Found CUDA libraries at ${CUDA_LIBRARIES}")

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{TensorRT_PATH}
        PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
include_directories(${TENSORRT_INCLUDE_DIR})
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{TensorRT_PATH}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{TensorRT_PATH}
        PATH_SUFFIXES lib lib64 lib/x64)
if(WIN32)
    find_library(TENSORRT_LIBRARY_MYELIN myelin64_1
            HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{TensorRT_PATH}
            PATH_SUFFIXES lib lib64 lib/x64)
else()
    find_library(TENSORRT_LIBRARY_MYELIN myelin
            HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{TensorRT_PATH}
            PATH_SUFFIXES lib lib64 lib/x64)
endif()
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_MYELIN})
MESSAGE(STATUS "Found TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
    message(FATAL_ERROR "Could not find TensorRT library.")
endif()

cuda_add_library(yoloplugin STATIC ../../tensorrtx/yolov5/yololayer.cu)
target_link_libraries(yoloplugin ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY})

add_executable(TrafficDetect TrafficDetect.cpp Core.cpp TrtEngine.cpp Yolo5Engine.cpp)
target_link_libraries(TrafficDetect yoloplugin ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY})
