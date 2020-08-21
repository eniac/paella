#include <llis/ipc/shared_memory_channel.h>

int main() {
    llis::ipc::SharedMemoryChannel channel("test");
    for (int i = 0; i < 10000; ++i) {
        channel.write(&i, sizeof(i));
    }
}

