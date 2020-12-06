#include <llis/job/coroutine_job.h>
#include <llis/job/context.h>
#include <llis/job/instrument.h>

#include <cstdio>

__global__ void helloworld(int i, void* job, llis::ipc::ShmChannelGpu gpu2sched_channel) {
    llis::job::kernel_start(job, &gpu2sched_channel);

    unsigned nsmid;
    asm("mov.u32 %0, %nsmid;" : "=r"(nsmid));
    printf("Hello world %d %d\n", i, nsmid);

    llis::job::kernel_end(job, &gpu2sched_channel);
}

class HelloWorldCoroutineJob : public llis::job::CoroutineJob {
  public:
    HelloWorldCoroutineJob() {
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
        CoroutineJob::full_init(io_ptr);

        io_ptr_ = io_ptr;
    }

    void body() override {
        for (int i = 0; i < 5; ++i) {
            ++num_;
            set_num_blocks(num_);

            yield();
            num_running_blocks_ = num_;
            helloworld<<<num_running_blocks_, 1, 0, get_cuda_stream()>>>(num_, this, llis::job::Context::get_gpu2sched_channel()->fork());
        }
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
    return new HelloWorldCoroutineJob();
}

}

