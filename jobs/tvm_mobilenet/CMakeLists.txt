add_library(job_tvm_mobilenet SHARED tvm_mobilenet.cpp)
target_link_libraries(job_tvm_mobilenet tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_mobilenet DESTINATION lib/llis_jobs)

