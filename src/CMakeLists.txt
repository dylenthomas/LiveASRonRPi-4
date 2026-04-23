cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(LARS LANGUAGES C CUDA)

set(PROJECT_SOURCE "/home/dylenthomas/LiveASRonRPi-4")
set(LIBS "${PROJECT_SOURCE}/libs")
set(ORT_LIB_DIR "${LIBS}/onnxruntime")
set(VENV_DIR ${CMAKE_BINARY_DIR}/venv)

find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Embed)
find_package(CUDAToolkit REQUIRED)

find_library(ORT_LIB onnxruntime PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH REQUIRED)
find_library(MIC_LIB mic_access PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(FFT_LIB fft PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)
find_library(ALSA_LIB asound REQUIRED)
find_library(CNFG_PARSR config_parser PATHS ${LIBS} NO_DEFAULT_PATH REQUIRED)

add_custom_target(create_venv
    COMMAND ${Python_EXECUTABLE} -m venv ${VENV_DIR}
    COMMENT "Creating virtual environment..."
)

add_custom_target(python_deps
    COMMAND ${VENV_DIR}/bin/pip install torch numpy
    COMMENT "Installing Python dependencies..."
    DEPENDS create_venv
)

add_executable(main main.c)

target_include_directories(main PRIVATE
    "${PROJECT_SOURCE}/include"
    "${PROJECT_SOURCE}/include/onnxruntime"
    "${VENV_DIR}/include/python3.12"
)

target_link_libraries(main PRIVATE
    ${ORT_LIB}
    ${ALSA_LIB}
    ${MIC_LIB}
    ${FFT_LIB}
    ${TRANSC_LIB}
    ${CNFG_PARSR}
    Python::Python
    ${Python_LIBRARIES}
    m
    CUDA::cudart
    CUDA::cublas
)

add_dependencies(main python_deps)

set_target_properties(main PROPERTIES 
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ORT_LIB_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
