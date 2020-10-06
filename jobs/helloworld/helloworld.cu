#include <llis/job.h>

#include <cstdio>

__global__ void helloworld(int i, void* job, llis::ipc::ShmChannelGpu gpu2sched_channel) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));

        gpu2sched_channel.acquire_writer_lock();
        gpu2sched_channel.write(true);
        gpu2sched_channel.write(job);
        gpu2sched_channel.write(smid);
        gpu2sched_channel.release_writer_lock();
    }

    unsigned nsmid;
    asm("mov.u32 %0, %nsmid;" : "=r"(nsmid));
    printf("Hello world %d %d\n", i, nsmid);

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        gpu2sched_channel.acquire_writer_lock();
        gpu2sched_channel.write(false);
        gpu2sched_channel.write(job);
        gpu2sched_channel.release_writer_lock();
    }
}

class HelloWorldJob : public llis::Job {
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
        helloworld<<<num_running_blocks_, 1, 0, get_cuda_stream()>>>(num_, this, gpu2sched_channel_.fork());
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

    unsigned get_num_blocks() override {
        return num_ + 1;
    }

    unsigned get_num_threads_per_block() override {
        return 1;
    }

    unsigned get_smem_size_per_block() override {
        return 0;
    }

    unsigned get_num_registers_per_thread() override {
        return 32;
    }

  private:
    void* io_ptr_;
    int num_ = 0;
    int num_running_blocks_;
};

extern "C" {

llis::Job* init_job() {
    return new HelloWorldJob();
}

}

