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

option(SCHED_FIFO "FIFO scheduler" OFF)
if(SCHED_FIFO)
    add_definitions(-DLLIS_SCHED_FIFO)
endif(SCHED_FIFO)

option(SCHED_FIFO2 "FIFO2 scheduler" OFF)
if(SCHED_FIFO2)
    add_definitions(-DLLIS_SCHED_FIFO2)
endif(SCHED_FIFO2)

option(SCHED_FULL2 "Full2 scheduler" OFF)
if(SCHED_FULL2)
    add_definitions(-DLLIS_SCHED_FULL2)
endif(SCHED_FULL2)

option(SCHED_FULL3 "Full3 scheduler" OFF)
if(SCHED_FULL3)
    add_definitions(-DLLIS_SCHED_FULL3)
endif(SCHED_FULL3)

option(ENABLE_PROFILER "Enable profiler" OFF)
if(ENABLE_PROFILER)
    add_definitions(-DLLIS_ENABLE_PROFILER)
endif(ENABLE_PROFILER)

include_directories(include)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(jobs)

