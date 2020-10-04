#include <llis/job.h>

#include <cstdio>

__global__ void run(int n, void* job, llis::ipc::ShmChannelGpu gpu2sched_channel) {
    printf("run_forever %p\n", job);

    if (n < 5) {
        volatile int i;
        for (i = 0; i < n * 1000; ++i);
    } else {
        while (true);
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        gpu2sched_channel.acquire_writer_lock();
        gpu2sched_channel.write(false);
        gpu2sched_channel.write(job);
        gpu2sched_channel.release_writer_lock();
    }
}

class RunForeverJob : public llis::Job {
  public:
    size_t get_input_size() override {
        return 5;
    }

    size_t get_output_size() override {
        return 11;
    }

    size_t get_param_size() override {
        return 4;
    }

    void full_init(void* io_ptr) override {
        io_ptr_ = io_ptr;
    }

    void run_next() override {
        ++num_;

        num_running_blocks_ = num_;
        run<<<num_running_blocks_, 1, 0, get_cuda_stream()>>>(num_, this, gpu2sched_channel_.fork());
    }

    bool has_next() const override {
        return num_ < 5;
    }

    void mark_block_finish() override {
        num_running_blocks_--;
        if (num_running_blocks_ == 0) {
            unset_running();
        }
    }

  private:
    void* io_ptr_;
    int num_ = 0;
    int num_running_blocks_;
};

extern "C" {

llis::Job* init_job() {
    return new RunForeverJob();
}

}


