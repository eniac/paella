add_subdirectory(latency)

add_executable(shmc_read shmc_read.cpp $<TARGET_OBJECTS:ipc>)
target_link_libraries(shmc_read rt)
add_executable(shmc_write shmc_write.cpp $<TARGET_OBJECTS:ipc>)
target_link_libraries(shmc_write rt)
add_executable(shmc_read_write shmc_read_write.cpp $<TARGET_OBJECTS:ipc>)
target_link_libraries(shmc_read_write rt)
