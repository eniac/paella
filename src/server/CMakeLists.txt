add_executable(server server.cpp scheduler.cpp client_connection.cpp registered_job.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(server rt dl)

