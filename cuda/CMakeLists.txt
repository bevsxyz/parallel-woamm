cmake_minimum_required(VERSION 3.16)
project(whale)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)

add_executable(whale
        src/main.cpp
        src/func/func.cu
        src/func/func.h
        src/runner.h
        src/runner.cpp
        src/data/data_stats.cpp
        src/data/data_stats.h
        src/oa/oa.cu
        src/oa/oa.h)