add_library(job_tvm_resnet18 SHARED tvm_resnet18.cpp)
target_link_libraries(job_tvm_resnet18 tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_resnet18 DESTINATION lib/llis_jobs)

