add_library(job_tvm_resnet34 SHARED tvm_resnet34.cpp)
target_link_libraries(job_tvm_resnet34 tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_resnet34 DESTINATION lib/llis_jobs)

