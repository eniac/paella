#add_executable(test_direct_run_job direct_run_job.cpp $<TARGET_OBJECTS:ipc>)
#target_link_libraries(test_direct_run_job llis_job_gpu llis_context CUDA::cudart dl rt)

add_executable(test_tvm_direct_concurrent tvm_direct_concurrent.cpp)
#set_source_files_properties(tvm_direct_concurrent.cpp PROPERTIES COMPILE_FLAGS "-g -O0")
target_include_directories(test_tvm_direct_concurrent PUBLIC "${TVM_PATH}/include" "${TVM_PATH}/3rdparty/dmlc-core/include" "${TVM_PATH}/3rdparty/dlpack/include")
target_link_directories(test_tvm_direct_concurrent PUBLIC "${TVM_PATH}/build")
target_link_libraries(test_tvm_direct_concurrent tvm_runtime pthread)

add_executable(test_tvm_direct_multistream tvm_direct_multistream.cpp)
#set_source_files_properties(tvm_direct_multistream.cpp PROPERTIES COMPILE_FLAGS "-g -O0")
target_include_directories(test_tvm_direct_multistream PUBLIC "${TVM_PATH}/include" "${TVM_PATH}/3rdparty/dmlc-core/include" "${TVM_PATH}/3rdparty/dlpack/include")
target_link_directories(test_tvm_direct_multistream PUBLIC "${TVM_PATH}/build")
#set_property(TARGET test_tvm_direct_multistream PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(test_tvm_direct_multistream PUBLIC "-fPIC" PUBLIC "-fPIE")
target_link_libraries(test_tvm_direct_multistream tvm_runtime CUDA::cudart pthread)

add_executable(cuda_sync_benchmark cuda_sync_benchmark.cu)
target_link_libraries(cuda_sync_benchmark CUDA::cudart dl rt)

add_executable(cuda_callback_benchmark cuda_callback_benchmark.cu)
target_link_libraries(cuda_callback_benchmark CUDA::cudart dl rt)

add_executable(test_mmap_mlock_limit mmap_mlock_limit.cpp)
target_link_libraries(test_mmap_mlock_limit rt)
