add_library(job_run_forever SHARED run_forever.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_run_forever PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

