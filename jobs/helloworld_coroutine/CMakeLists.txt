add_library(job_helloworld_coroutine SHARED helloworld_coroutine.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job_helloworld_coroutine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(job_helloworld_coroutine llis_job_gpu llis_context)
