#include <llis/ipc/shm_channel_1to1.h>

int main() {
    llis::ipc::ShmChannel1to1 channel("test");
    for (int i = 0; i < 10000; ++i) {
        channel.write(&i, sizeof(i));
    }
}

