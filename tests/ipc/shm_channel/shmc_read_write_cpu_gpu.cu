#include <llis/ipc/shm_channel.h>

#include <thread>

void reader(llis::ipc::ShmChannelGpu* channel) {
    for (int i = 0; i < 10000; ++i) {
        int val;
        channel->read(&val, sizeof(val));
        if (val != i) {
            printf("Error! Expected: %d, Actual: %d\n", i, val);
            break;
        }
    }
}

__global__ void writer(llis::ipc::ShmChannelGpu channel) {
    for (int i = 0; i < 10000; ++i) {
        channel.write(i);
    }
}

int main() {
    llis::ipc::ShmChannelGpu channel(64);
    llis::ipc::ShmChannelGpu channel_gpu(&channel);

    std::thread reader_thr(reader, &channel);

    writer<<<1, 1>>>(std::move(channel_gpu));

    reader_thr.join();
    cudaDeviceSynchronize();
}

