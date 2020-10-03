add_library(job_helloworld SHARED helloworld.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_helloworld PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
