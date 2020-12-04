#include <llis/ipc/shm_channel.h>
#include <llis/job/instrument.h>

#include <chrono>
#include <iostream>

__global__ void helloworld(int i, void* job, llis::ipc::ShmChannelGpu gpu2sched_channel) {
    llis::job::kernel_start(job, &gpu2sched_channel);
    llis::job::kernel_end(job, &gpu2sched_channel);
}

int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    llis::ipc::ShmChannelGpu gpu2sched_channel(1024);

    for (int i = 0; i < 10; ++i) {
        auto start_time = std::chrono::steady_clock::now();

        helloworld<<<1, 1, 0, stream>>>(i, &start_time, gpu2sched_channel.fork());
        cudaStreamSynchronize(stream);

        auto end_time = std::chrono::steady_clock::now();

        auto time_taken = end_time - start_time;
        std::cout << std::chrono::duration<double, std::micro>(time_taken).count() << std::endl;
    }
}

