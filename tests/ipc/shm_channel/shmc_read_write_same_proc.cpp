#include <llis/ipc/shm_channel.h>

#include <thread>

void reader(llis::ipc::ShmChannelCpuReader* channel) {
    for (int i = 0; i < 10000; ++i) {
        int val;
        channel->read(&val, sizeof(val));
        if (val != i) {
            printf("Error! Expected: %d, Actual: %d\n", i, val);
            break;
        }
    }
}

void writer(llis::ipc::ShmChannelCpuWriter* channel) {
    for (int i = 0; i < 10000; ++i) {
        channel->write(i);
    }
}

int main() {
    llis::ipc::ShmChannelCpuReader channelRead(64);
    llis::ipc::ShmChannelCpuWriter channelWrite = channelRead.fork();

    std::thread reader_thr(reader, &channelRead);
    std::thread writer_thr(writer, &channelWrite);

    reader_thr.join();
    writer_thr.join();
}

