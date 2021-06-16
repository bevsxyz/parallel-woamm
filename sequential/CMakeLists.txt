cmake_minimum_required(VERSION 3.6)
project(whale)

set(CMAKE_CXX_STANDARD 11)

add_executable(whale
        src/main.cpp
        lib/mt64.h
        src/func/func.cpp
        src/func/func.h
        src/population/population.h
        src/population/population.cpp
        src/runner.h
        src/runner.cpp
        src/data/data_stats.cpp
        src/data/data_stats.h
        src/whale/woam.cpp
        src/whale/woam.h)
