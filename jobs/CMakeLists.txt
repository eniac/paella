add_subdirectory(helloworld)
add_subdirectory(helloworld_coroutine)
add_subdirectory(run_forever)
add_subdirectory(cnn)

if(TVM_PATH)
    add_subdirectory(tvm)
endif(TVM_PATH)

