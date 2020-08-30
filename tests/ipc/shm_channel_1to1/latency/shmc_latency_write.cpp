#include <llis/ipc/shm_channel_1to1.h>

#include <iostream>
#include <chrono>

int main() {
    int val = 1234;
    int val2 = 5678;

    llis::ipc::ShmChannel1to1 channel("test", sizeof(val) + 1);

    channel.write(&val2, sizeof(val2));


    channel.write(&val, sizeof(val));
    std::cout << "Current time since epoch: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
}

