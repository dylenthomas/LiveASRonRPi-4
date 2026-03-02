cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(OnnxGpuTest LANGUAGES C CUDA)

set(PROJECT_SOURCE "/home/dylenthomas/LiveASRonRPi-4")
set(LIBS "${PROJECT_SOURCE}/libs")
set(ORT_LIB_DIR "${LIBS}/onnxruntime")

find_package(Python 3.12 EXACT REQUIRED COMPONENTS Development.Embed)

find_library(ORT_LIB onnxruntime PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(MIC_LIB mic_access PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(FFT_LIB fft PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(ALSA_LIB asound REQUIRED)
find_library(TRANSC_LIB transcripter.cpython-312 PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(CNFG_PARSR config_parser PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)

add_executable(main main.c)

include_directories(main PRIVATE
    "${PROJECT_SOURCE}/include"
    "${PROJECT_SOURCE}/include/onnxruntime"
    "/usr/include/python3.12"
)

target_link_libraries(main PRIVATE
    ${ORT_LIB}
    ${ALSA_LIB}
    ${MIC_LIB}
    ${FFT_LIB}
    ${TRANSC_LIB}
    ${CNFG_PARSR}
    Python::Python
    m

)

set_target_properties(main PROPERTIES 
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ORT_LIB_DIR}:${LOCAL_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
