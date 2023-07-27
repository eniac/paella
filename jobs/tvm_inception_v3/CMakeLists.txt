add_library(job_tvm_inception_v3 SHARED tvm_inception_v3.cpp)
target_link_libraries(job_tvm_inception_v3 tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_inception_v3 DESTINATION lib/llis_jobs)

