add_library(job_tvm_squeezenet1_1 SHARED tvm_squeezenet1_1.cpp)
target_link_libraries(job_tvm_squeezenet1_1 tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_squeezenet1_1 DESTINATION lib/llis_jobs)

