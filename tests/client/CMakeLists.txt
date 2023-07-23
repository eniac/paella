add_executable(test_client client.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(test_client spdlog::spdlog dl rt)

add_executable(test_client_concurrent_runs client_concurrent_runs.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(test_client_concurrent_runs spdlog::spdlog dl rt)

add_executable(test_client_single_latency client_single_latency.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(test_client_single_latency spdlog::spdlog dl rt)

add_executable(test_client_concurrent_run_latencies client_concurrent_run_latencies.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(test_client_concurrent_run_latencies spdlog::spdlog dl rt)

add_executable(test_client_concurrent_run_latencies_set_load client_concurrent_run_latencies_set_load.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(test_client_concurrent_run_latencies_set_load spdlog::spdlog dl rt)

add_executable(test_client_concurrent_run_latencies_set_load_multi client_concurrent_run_latencies_set_load_multi.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(test_client_concurrent_run_latencies_set_load_multi spdlog::spdlog dl rt)

add_executable(test_raw_kernel_launch raw_kernel_launch.cu $<TARGET_OBJECTS:ipc-gpu>)
target_link_libraries(test_raw_kernel_launch llis_job dl rt)
set_target_properties(test_raw_kernel_launch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
