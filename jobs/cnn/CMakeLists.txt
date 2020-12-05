add_library(job_cnn SHARED main.cu layer.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_cnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(job_cnn llis_job_gpu llis_context)
