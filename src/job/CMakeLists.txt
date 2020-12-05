add_library(llis_job SHARED coroutine_job.cpp $<TARGET_OBJECTS:ipc>)
set_target_properties(llis_job PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(llis_job Boost::context)

add_library(llis_job_gpu SHARED utils.cu coroutine_job.cpp $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(llis_job_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(llis_job_gpu Boost::context)

add_library(llis_context SHARED context.cpp)

