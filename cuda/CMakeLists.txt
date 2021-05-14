cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)


include(CTest)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")


include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

add_library(whale STATIC
        src/oa/oa.cu
        src/func/func.cu
        src/runner.cu
        src/data/data_stats.cpp
        )

target_compile_features(whale PUBLIC cxx_std_11)


set_target_properties(whale
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(whale_test src/main.cu)

set_property(TARGET whale_test 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(whale_test whale)

