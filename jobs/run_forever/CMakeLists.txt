add_library(job_run_forever SHARED run_forever.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_run_forever PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(job_run_forever llis_job_gpu llis_context)

