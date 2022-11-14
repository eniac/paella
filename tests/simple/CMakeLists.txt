#add_executable(test_direct_run_job direct_run_job.cpp $<TARGET_OBJECTS:ipc>)
#target_link_libraries(test_direct_run_job llis_job_gpu llis_context CUDA::cudart dl rt)

add_executable(test_tvm_direct_concurrent tvm_direct_concurrent.cpp)
#set_source_files_properties(tvm_direct_concurrent.cpp PROPERTIES COMPILE_FLAGS "-g -O0")
target_include_directories(test_tvm_direct_concurrent PUBLIC "${TVM_PATH}/include" "${TVM_PATH}/3rdparty/dmlc-core/include" "${TVM_PATH}/3rdparty/dlpack/include")
target_link_directories(test_tvm_direct_concurrent PUBLIC "${TVM_PATH}/build")
target_link_libraries(test_tvm_direct_concurrent tvm_runtime pthread)