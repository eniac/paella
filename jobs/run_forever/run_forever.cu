#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/job.h>
#include <llis/job/instrument.h>
#include <llis/job/context.h>

#include <cstdio>

__global__ void run(int n, llis::JobId job_id, llis::ipc::Gpu2SchedChannel gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
        , llis::ipc::Gpu2SchedChannel gpu2sched_block_time_channel
#endif
) {
#ifdef LLIS_MEASURE_BLOCK_TIME
    llis::job::BlockStartEndTime start_end_time;
    llis::job::kernel_start(job_id, &gpu2sched_channel, &start_end_time);
#else
    llis::job::kernel_start(job_id, &gpu2sched_channel);
#endif

    printf("run_forever %u\n", job_id);

    if (n < 5) {
        volatile int i;
        for (i = 0; i < n * 1000; ++i);
    } else {
        while (true);
    }

#ifdef LLIS_MEASURE_BLOCK_TIME
    llis::job::kernel_end(job_id, &gpu2sched_channel, &gpu2sched_block_time_channel, &start_end_time);
#else
    llis::job::kernel_end(job_id, &gpu2sched_channel);
#endif
}

class RunForeverJob : public llis::job::Job {
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

        num_ = 0;

        set_num_blocks(1);
        set_num_threads_per_block(1);
        set_smem_size_per_block(0);
        set_num_registers_per_thread(32);
    }

    void init(void* io_ptr) override {
        io_ptr_ = io_ptr;

        num_ = 0;

        set_num_blocks(1);
    }

    void run_next() override {
        ++num_;

        run<<<num_, 1, 0, get_cuda_stream()>>>(num_, get_id(), llis::job::Context::get_gpu2sched_channel()->fork()
#ifdef LLIS_MEASURE_BLOCK_TIME
                , llis::job::Context::get_gpu2sched_block_time_channel()->fork()
#endif
                );

        set_num_blocks(num_ + 1);
    }

    bool has_next() const override {
        return num_ < 5;
    }

  private:
    void* io_ptr_;
    int num_ = 0;
};

extern "C" {

llis::job::Job* init_job() {
    return new RunForeverJob();
}

}


