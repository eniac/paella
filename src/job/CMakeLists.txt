add_library(job-gpu OBJECT utils.cu $<TARGET_OBJECTS:ipc-gpu>)
set_target_properties(job-gpu PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)

