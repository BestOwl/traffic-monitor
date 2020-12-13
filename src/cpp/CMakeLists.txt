find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA Toolkit
if (NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include
        )
include_directories(${CUDA_INCLUDE_DIR})
find_library(CUDA_LIBRARIES
        NAMES cudart
        HINTS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/Win32 lib/import lib64/import
        PATHS ${CUBLAS_PATHS}
        DOC "CUDA library"
        )
MESSAGE(STATUS "Found CUDA libraries at ${CUDA_LIBRARIES}")

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
include_directories(${TENSORRT_INCLUDE_DIR})
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
if(WIN32)
    find_library(TENSORRT_LIBRARY_MYELIN myelin64_1
            HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)
else()
    find_library(TENSORRT_LIBRARY_MYELIN myelin
            HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)
endif()
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_MYELIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
    message(ERROR "Cannot find TensorRT library.")
endif()

SET(DeepStreamSDK_Include_DIR "/opt/nvidia/deepstream/deepstream/sources/includes")
MESSAGE(STATUS "${DeepStreamSDK_Include_Dir}")
include_directories(${DeepStreamSDK_Include_DIR})

#SET(NvDsInfer_Sources_DIR "/opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer")
#SET(NvDsInfer_Sources
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_backend.cpp
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_backend.h
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_context_impl.cpp
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_context_impl.h
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_context_impl_capi.cpp
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_context_impl_output_parsing.cpp
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_conversion.h
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_func_utils.cpp
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_model_builder.cpp
#        ${NvDsInfer_Sources_DIR}/nvdsinfer_model_builder.h)

SET(DeepStreamSDK_Lib_DIR "/opt/nvidia/deepstream/deepstream/lib")
SET(NvDsInfer_Libs
        ${DeepStreamSDK_Lib_DIR}/libnvds_infer.so)

add_executable(TrtTrafficDetect TrtTrafficDetect.cpp DetectNetEngine.cpp DetectNetEngine.h)
target_link_libraries(TrtTrafficDetect ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TENSORRT_LIBRARY} ${NvDsInfer_Libs})