add_library(job_dummy_short SHARED dummy_short.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_dummy_short PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_options(job_dummy_short PRIVATE -fvisibility=hidden)
target_link_libraries(job_dummy_short llis_job_gpu llis_context)
