add_library(job_cnn SHARED main.cu layer.cu $<TARGET_OBJECTS:job-gpu> $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_cnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
