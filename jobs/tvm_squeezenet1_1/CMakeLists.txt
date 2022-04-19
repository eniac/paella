add_library(job_tvm_squeezenet1_1 SHARED tvm_squeezenet1_1.cpp)
target_include_directories(job_tvm_squeezenet1_1 PUBLIC "${TVM_PATH}/include" "${TVM_PATH}/3rdparty/dmlc-core/include" "${TVM_PATH}/3rdparty/dlpack/include")
target_link_directories(job_tvm_squeezenet1_1 PUBLIC "${TVM_PATH}/build")
target_link_libraries(job_tvm_squeezenet1_1 llis_job llis_context tvm_runtime)

