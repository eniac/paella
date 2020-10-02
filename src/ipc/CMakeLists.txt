add_library(ipc OBJECT shm_channel.cu)

add_library(ipc-gpu OBJECT shm_channel.cu)
set_target_properties(ipc-gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
