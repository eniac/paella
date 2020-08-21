#include <llis/ipc/shared_memory_channel.h>

#include <iostream>
#include <chrono>

int main() {
    int val = 1234;
    int val2 = 5678;

    llis::ipc::SharedMemoryChannel channel("test", sizeof(val) + 1);

    channel.write(&val2, sizeof(val2));


    channel.write(&val, sizeof(val));
    std::cout << "Current time since epoch: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
}

