add_library(job_tvm_mnist SHARED tvm_mnist.cpp)
target_link_libraries(job_tvm_mnist tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_mnist DESTINATION lib/llis_jobs)

