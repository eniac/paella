add_library(llis_job SHARED finished_block_notifier.cu $<TARGET_OBJECTS:ipc>)
set_target_properties(llis_job PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(llis_job "-Wl,--no-as-needed" Boost::context "-Wl,--as-needed")

add_library(llis_job_gpu SHARED finished_block_notifier.cu utils.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(llis_job_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(llis_job_gpu Boost::context)

add_library(llis_context SHARED context.cpp)
target_link_libraries(llis_context llis_job)
