#define CHANNEL_SIZE 64
#define NUM_ITERS 1000000

#include <llis/ipc/shm_channel.h>

#include <iostream>
#include <chrono>

int main() {
    int val = 0;

    llis::ipc::ShmChannelCpuReader read_channel("test_read", 64);
    llis::ipc::ShmChannelCpuWriter write_channel("test_write", 64);

    // Warm up
    write_channel.write(&val, sizeof(val));
    write_channel.write(&val, sizeof(val));
    read_channel.read(&val, sizeof(val));
    read_channel.read(&val, sizeof(val));

    auto time1 = std::chrono::system_clock::now().time_since_epoch().count();

    for (int i = 1; i < NUM_ITERS; ++i) {
        read_channel.read(&val, sizeof(val));
        write_channel.write(&i, sizeof(i));
    }
    read_channel.read(&val, sizeof(val));

    auto time2 = std::chrono::system_clock::now().time_since_epoch().count();

    std::cout << "time1: " << time1 << std::endl;
    std::cout << "time2: " << time2 << std::endl;
}

