#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <vector>

__global__ void dummy_cuda_sync() {
}

__global__ void dummy_flag(volatile int* flag) {
    __threadfence_system();
    *flag = 1;
}

void run_cuda_sync(int num_iter, cudaStream_t stream) {
    for (int i = 0; i < num_iter; ++i) {
        dummy_cuda_sync<<<1, 1, 0, stream>>>();
        cudaStreamSynchronize(stream);
    }
}

void run_flag(int num_iter, cudaStream_t stream, volatile int* flag) {
    for (int i = 0; i < num_iter; ++i) {
        *flag = 0;

        dummy_flag<<<1, 1, 0, stream>>>(flag);

        while (*flag == 0);
    }
}

int main(int argc, char** argv) {
    int num_iter = atoi(argv[1]);
    int num_thrs = atoi(argv[2]);

    std::vector<cudaStream_t> streams;
    streams.resize(num_thrs);
    for (int i = 0; i < num_thrs; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    volatile int* flags;
    cudaMallocHost(&flags, sizeof(int) * num_thrs);

    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    double time_cuda_sync;
    double time_flag;

    {
    std::vector<std::thread> thrs;
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < num_thrs; ++i) {
        thrs.emplace_back(run_cuda_sync, num_iter, streams[i]);
    }
    for (auto& thr : thrs) {
        thr.join();
    }
    auto end_time = std::chrono::steady_clock::now();
    time_cuda_sync = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    }

    {
    std::vector<std::thread> thrs;
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < num_thrs; ++i) {
        thrs.emplace_back(run_flag, num_iter, streams[i], flags + i);
    }
    for (auto& thr : thrs) {
        thr.join();
    }
    auto end_time = std::chrono::steady_clock::now();
    time_flag = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    }

    printf("%f,%f\n", time_cuda_sync, time_flag);
}

