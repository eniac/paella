cmake_minimum_required(VERSION 3.5)

project(llis LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

find_package(Boost REQUIRED context)

include_directories(include)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(jobs)

