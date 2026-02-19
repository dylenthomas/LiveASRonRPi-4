cmake_minimum_required(VERSION 3.18)
project(OnnxGpuTest LANGUAGES C CUDA)

set(PROJECT_SOURCE "/home/dylenthomas/Calliope/LARS/LiveASRonRPi-4")
set(LIBS "${PROJECT_SOURCE}/libs")
set(ORT_LIB_DIR "${LIBS}/onnxruntime-linux-x64-gpu-1.23.2")

find_library(ORT_LIB onnxruntime PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(MIC_LIB mic_access PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(ALSA_LIB asound REQUIRED)

add_executable(test onnxrt_vad_gpu_test.c)

include_directories(test PRIVATE
    "${PROJECT_SOURCE}/include"
    "${PROJECT_SOURCE}/include/onnxruntime-linux-x64-gpu-1.23.2"
)


target_link_libraries(test PRIVATE
    ${ORT_LIB}
    ${ALSA_LIB}
    ${MIC_LIB} 
)

set_target_properties(test PROPERTIES 
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ORT_LIB_DIR}:${LOCAL_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
