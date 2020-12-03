add_library(ipc OBJECT shm_channel.cu name_format.cpp)

add_library(ipc-gpu OBJECT shm_channel.cu name_format.cpp)
set_target_properties(ipc-gpu PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)
