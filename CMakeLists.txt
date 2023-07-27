cmake_minimum_required(VERSION 3.5)

project(llis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0105 NEW)

find_package(CUDAToolkit)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

find_package(Boost REQUIRED system context program_options)

find_package(spdlog REQUIRED)

option(MEASURE_BLOCK_TIME "Enable measurement of block time" OFF)
if(MEASURE_BLOCK_TIME)
    add_definitions(-DLLIS_MEASURE_BLOCK_TIME)
endif(MEASURE_BLOCK_TIME)

option(FINISHED_BLOCK_NOTIFICATION_AGG "Aggregate block notifications" ON)
if(FINISHED_BLOCK_NOTIFICATION_AGG)
    add_definitions(-DLLIS_FINISHED_BLOCK_NOTIFICATION_AGG)
endif(FINISHED_BLOCK_NOTIFICATION_AGG)

option(ENABLE_PROFILER "Enable profiler" OFF)
if(ENABLE_PROFILER)
    add_definitions(-DLLIS_ENABLE_PROFILER)
endif(ENABLE_PROFILER)

option(PRINT_LAUNCH_JOB_IPC_LATENCY "Print launch job IPC latency" OFF)
if(PRINT_LAUNCH_JOB_IPC_LATENCY)
    add_definitions(-DPRINT_LAUNCH_JOB_IPC_LATENCY)
endif(PRINT_LAUNCH_JOB_IPC_LATENCY)

set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/../lib)

include_directories(include)

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(jobs EXCLUDE_FROM_ALL)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

