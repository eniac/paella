add_library(ipc OBJECT shm_channel.cu shm_primitive_channel.cu name_format.cpp unix_datagram_socket.cpp)
set_target_properties(ipc PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ipc-gpu OBJECT shm_channel.cu shm_primitive_channel.cu name_format.cpp unix_datagram_socket.cpp)
set_target_properties(ipc-gpu PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)
