add_library(job_tvm_ultraface320 SHARED tvm_ultraface320.cpp)
target_link_libraries(job_tvm_ultraface320 tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_ultraface320 DESTINATION lib/llis_jobs)

