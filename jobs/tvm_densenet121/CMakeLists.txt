add_library(job_tvm_densenet121 SHARED tvm_densenet121.cpp)
target_link_libraries(job_tvm_densenet121 tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_densenet121 DESTINATION lib/llis_jobs)

