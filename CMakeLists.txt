cmake_minimum_required(VERSION 3.10)
project(torch_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE ATEN_SRC "aten/src/*.cpp")
file(GLOB_RECURSE TORCH_SRC "torch/src/*.cpp")

add_executable(main ${ATEN_SRC} ${TORCH_SRC} main.cpp)

target_include_directories(main PRIVATE aten/include torch/include)