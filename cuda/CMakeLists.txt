cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)


include(CTest)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

add_library(whale STATIC
        lib/mpgp32.cu
        src/main.cpp
        src/oa/oa.cu
        src/oa/oa.h
        src/func/func.cu
        src/func/func.h
        src/runner.h
        src/runner.cpp
        src/data/data_stats.cpp
        src/data/data_stats.h
        )

target_compile_features(whale PUBLIC cxx_std_11)


set_target_properties(whale
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(whale_test src/main.cpp)

set_property(TARGET whale_test 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(whale_test whale)

