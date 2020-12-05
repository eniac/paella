add_library(job_helloworld SHARED helloworld.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_helloworld PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(job_helloworld llis_job_gpu llis_context)
