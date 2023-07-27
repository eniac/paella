#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <atomic>

std::vector<cudaStream_t> streams;
std::queue<unsigned> noti_queue;
std::atomic_uint noti_queue_num;
std::mutex mtx;

__global__ void dummy() {
}

void callback(void* stream_id_) {
    unsigned stream_id = (unsigned long)stream_id_;
    std::unique_lock<std::mutex> lock(mtx);
    noti_queue.push(stream_id);
    lock.unlock();
    noti_queue_num.fetch_add(1, std::memory_order_release);
}

int main(int argc, char** argv) {
    unsigned num_iter = atoi(argv[1]);
    unsigned num_streams = atoi(argv[2]);

    streams.resize(num_streams);
    for (unsigned i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    std::vector<unsigned> streams_finished;
    streams_finished.resize(num_streams);

    unsigned total_finished = 0;
    const unsigned total_num = num_iter * num_streams;

    noti_queue_num.store(0);

    auto start_time = std::chrono::steady_clock::now();

    for (unsigned i = 0; i < num_streams; ++i) {
        dummy<<<1, 1, 0, streams[i]>>>();
        cudaLaunchHostFunc(streams[i], callback, (void*)i);
    }

    while (total_finished < total_num) {
        while (noti_queue_num.load(std::memory_order_acquire) == 0);
        noti_queue_num.fetch_sub(1, std::memory_order_release);

        std::unique_lock<std::mutex> lock(mtx);
        unsigned stream_id = noti_queue.front();
        noti_queue.pop();
        lock.unlock();

        ++streams_finished[stream_id];
        ++total_finished;

        if (streams_finished[stream_id] < num_iter) {
            dummy<<<1, 1, 0, streams[stream_id]>>>();
            cudaLaunchHostFunc(streams[stream_id], callback, (void*)stream_id);
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    double time_elasped = std::chrono::duration<double, std::micro>(end_time - start_time).count();

    printf("%f\n", time_elasped);
}

