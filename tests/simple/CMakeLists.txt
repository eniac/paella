#add_executable(test_direct_run_job direct_run_job.cpp $<TARGET_OBJECTS:ipc>)
#target_link_libraries(test_direct_run_job llis_job_gpu llis_context CUDA::cudart dl rt)

if(tvm_FOUND)
    add_executable(tvm_direct_concurrent tvm_direct_concurrent.cpp)
    target_link_libraries(tvm_direct_concurrent tvm::tvm_runtime llis_context CUDA::cudart pthread)

    add_executable(tvm_direct_multistream tvm_direct_multistream.cpp)
    target_link_libraries(tvm_direct_multistream tvm::tvm_runtime llis_context CUDA::cudart pthread)
    target_compile_definitions(tvm_direct_multistream PRIVATE SUBMIT_DIS)
    install(TARGETS tvm_direct_multistream DESTINATION bin)

    add_executable(tvm_direct_multistream_pregen tvm_direct_multistream.cpp)
    target_link_libraries(tvm_direct_multistream_pregen tvm::tvm_runtime llis_context CUDA::cudart pthread)
    target_compile_definitions(tvm_direct_multistream_pregen PRIVATE SUBMIT_PREGEN)
    target_compile_options(tvm_direct_multistream_pregen PUBLIC "-fPIC" PUBLIC "-fPIE")
    install(TARGETS tvm_direct_multistream_pregen DESTINATION bin)
endif(tvm_FOUND)

add_executable(cuda_sync_benchmark cuda_sync_benchmark.cu)
target_link_libraries(cuda_sync_benchmark CUDA::cudart dl rt)

add_executable(cuda_callback_benchmark cuda_callback_benchmark.cu)
target_link_libraries(cuda_callback_benchmark CUDA::cudart dl rt)

add_executable(test_mmap_mlock_limit mmap_mlock_limit.cpp)
target_link_libraries(test_mmap_mlock_limit rt)
