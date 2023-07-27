add_library(job_tvm_googlenet SHARED tvm_googlenet.cpp)
target_link_libraries(job_tvm_googlenet tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_googlenet DESTINATION lib/llis_jobs)

