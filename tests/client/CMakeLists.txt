add_executable(test_client client.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(test_client dl rt)

add_executable(test_client_concurrent_runs client_concurrent_runs.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(test_client_concurrent_runs dl rt)

add_executable(test_client_single_latency client_single_latency.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(test_client_single_latency dl rt)
