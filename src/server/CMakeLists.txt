add_executable(server server.cpp scheduler.cpp scheduler_fifo.cpp scheduler_fifo2.cpp scheduler_full3.cpp client_connection.cpp registered_job.cpp gpu_resources.cpp sm_resources.cpp profiler.cpp $<TARGET_OBJECTS:client> $<TARGET_OBJECTS:ipc>)
target_link_libraries(server llis_job llis_context spdlog::spdlog Boost::program_options rt dl)
