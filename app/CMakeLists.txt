add_executable(llis_app_client client.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(llis_app_client spdlog::spdlog dl rt)
install(TARGETS llis_app_client DESTINATION bin)
