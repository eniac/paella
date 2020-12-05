#include <llis/job/job.h>
#include <llis/job/instrument.h>
#include <llis/job/context.h>

#include <cstdio>

__global__ void run(int n, void* job, llis::ipc::ShmChannelGpu gpu2sched_channel) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));

        gpu2sched_channel.acquire_writer_lock();
        gpu2sched_channel.write(true);
        gpu2sched_channel.write(job);
        gpu2sched_channel.write(smid);
        gpu2sched_channel.release_writer_lock();
    }

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

class RunForeverJob : public llis::job::Job {
  public:
    RunForeverJob() {
        set_num_blocks(1);
        set_num_threads_per_block(1);
        set_smem_size_per_block(0);
        set_num_registers_per_thread(32);
    }

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
        run<<<num_running_blocks_, 1, 0, get_cuda_stream()>>>(num_, this, llis::job::Context::get_gpu2sched_channel()->fork());

        set_num_blocks(num_ + 1);
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

llis::job::Job* init_job() {
    return new RunForeverJob();
}

}


