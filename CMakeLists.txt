cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(predict_test predict_test.cpp)
target_link_libraries(predict_test "${TORCH_LIBRARIES}")
set_property(TARGET predict_test PROPERTY CXX_STANDARD 17)