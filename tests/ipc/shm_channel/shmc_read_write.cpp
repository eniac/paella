#include <llis/ipc/shm_channel.h>

int main() {
    llis::ipc::ShmChannelCpuReader channelRead("test");
    llis::ipc::ShmChannelCpuWriter channelWrite = channelRead.fork();
    for (int i = 0; i < 10000; ++i) {
        channelWrite.write(&i, sizeof(i));
        int val = -1;
        channelRead.read(&val, sizeof(val));
        if (val != i) {
            printf("Error! Expected: %d, Actual: %d\n", i, val);
            break;
        }
    }
}


