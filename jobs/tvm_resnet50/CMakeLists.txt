add_library(job_tvm_resnet50 SHARED tvm_resnet50.cpp)
target_link_libraries(job_tvm_resnet50 tvm::tvm_runtime llis_job llis_context)
install(TARGETS job_tvm_resnet50 DESTINATION lib/llis_jobs)

