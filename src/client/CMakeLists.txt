add_library(client OBJECT client.cpp job_ref.cpp job_instance_ref.cpp profiler_client.cpp)
target_link_libraries(client spdlog::spdlog)
