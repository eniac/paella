add_library(job_tvm_mnist SHARED tvm_mnist.cpp)
target_include_directories(job_tvm_mnist PUBLIC "${TVM_PATH}/include" "${TVM_PATH}/3rdparty/dmlc-core/include" "${TVM_PATH}/3rdparty/dlpack/include")
target_link_directories(job_tvm_mnist PUBLIC "${TVM_PATH}/build")
target_link_libraries(job_tvm_mnist llis_job llis_context tvm_runtime)

