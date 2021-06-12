#include <llis/ipc/shm_channel.h>

int main() {
    llis::ipc::ShmChannelCpuWriter channel("test");
    for (int i = 0; i < 10000; ++i) {
        channel.write(&i, sizeof(i));
    }
}

