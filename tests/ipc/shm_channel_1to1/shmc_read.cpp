#include <llis/ipc/shm_channel_1to1.h>

int main() {
    llis::ipc::ShmChannel1to1 channel("test", 64);
    for (int i = 0; i < 10000; ++i) {
        int val;
        channel.read(&val, sizeof(val));
        if (val != i) {
            printf("Error! Expected: %d, Actual: %d\n", i, val);
            break;
        }
    }
}

