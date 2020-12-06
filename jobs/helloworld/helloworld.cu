#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/job.h>
#include <llis/job/context.h>
#include <llis/job/instrument.h>

#include <cstdio>

__global__ void helloworld(int i, llis::JobId job_id, llis::ipc::Gpu2SchedChannel gpu2sched_channel) {
    llis::job::kernel_start(job_id, &gpu2sched_channel);

    unsigned nsmid;
    asm("mov.u32 %0, %nsmid;" : "=r"(nsmid));
    printf("Hello world %d %d\n", i, nsmid);

    llis::job::kernel_end(job_id, &gpu2sched_channel);
}

class HelloWorldJob : public llis::job::Job {
  public:
    HelloWorldJob() {
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

        helloworld<<<num_, 1, 0, get_cuda_stream()>>>(num_, get_id(), llis::job::Context::get_gpu2sched_channel()->fork());

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
    return new HelloWorldJob();
}

}

