add_library(job_dummy_11 SHARED dummy.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_dummy_11 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_options(job_dummy_11 PRIVATE -fvisibility=hidden)
target_link_libraries(job_dummy_11 llis_job_gpu llis_context)
