cmake_minimum_required(VERSION 3.5)

project(llis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

find_package(Boost REQUIRED context)

option(MEASURE_BLOCK_TIME "Enable measurement of block time" OFF)
if(MEASURE_BLOCK_TIME)
    add_definitions(-DLLIS_MEASURE_BLOCK_TIME)
endif(MEASURE_BLOCK_TIME)

include_directories(include)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(jobs)

