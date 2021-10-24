add_library(job_dummy_long SHARED dummy_long.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_dummy_long PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_options(job_dummy_long PRIVATE -fvisibility=hidden)
target_link_libraries(job_dummy_long llis_job_gpu llis_context)
