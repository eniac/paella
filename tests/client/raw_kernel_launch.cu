#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/instrument.h>
#include <llis/job/finished_block_notifier.h>
#include <llis/ipc/defs.h>

#include <chrono>
#include <iostream>

__global__ void helloworld(int i, llis::JobId job_id, llis::job::FinishedBlockNotifier* notifier) {
    //notifier->start(job_id);
    //notifier->end(job_id);
}

int main(int argc, char** argv) {
    int num_blocks = atoi(argv[1]);
    int num_iters = atoi(argv[2]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    llis::ipc::Gpu2SchedChannel gpu2sched_channel(1024);
    llis::ipc::Gpu2SchedChannel gpu2sched_block_time_channel(1024);

    llis::job::FinishedBlockNotifier* finished_block_notifier = llis::job::FinishedBlockNotifier::create_array(1, &gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
        , &gpu2sched_block_time_channel
#endif
    );

    for (int i = 0; i < num_iters; ++i) {
        auto start_time = std::chrono::steady_clock::now();

        helloworld<<<num_blocks, 1, 0, stream>>>(i, 0, finished_block_notifier);
        cudaStreamSynchronize(stream);

        auto end_time = std::chrono::steady_clock::now();

        auto time_taken = end_time - start_time;
        std::cout << std::chrono::duration<double, std::micro>(time_taken).count() << std::endl;
    }
}

