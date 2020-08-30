#include <llis/ipc/shm_channel_1to1.h>

#include <iostream>
#include <chrono>

int main() {
    int val;

    // The channel has a size of sizeof(val) + 1
    // This makes sure that the writer can only write after the reader has read
    // +1 because the writer always wastes one byte
    llis::ipc::ShmChannel1to1 channel("test", sizeof(val) + 1);

    // The first read is a barrier to make sure that both sides are in the same stage
    channel.read(&val, sizeof(val));

    channel.read(&val, sizeof(val));

    std::cout << "Current time since epoch: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;

    std::cout << "Value: " << val << std::endl;
}

