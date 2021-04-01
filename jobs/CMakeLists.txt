add_subdirectory(helloworld)
add_subdirectory(helloworld_coroutine)
add_subdirectory(vec_add_coroutine)
add_subdirectory(run_forever)
# Have to fix it before running it
#add_subdirectory(cnn)

if(TVM_PATH)
    add_subdirectory(tvm_mnist)
    add_subdirectory(tvm_mobilenet)
    add_subdirectory(tvm_resnet18)
endif(TVM_PATH)

