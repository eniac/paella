#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/coroutine_job.h>
#include <llis/job/context.h>
#include <llis/job/instrument.h>

#include <cstdio>

__global__ void dummy_kernel(float* mem, unsigned count, unsigned compute_count, llis::JobId job_id, llis::job::FinishedBlockNotifier* notifier) {
    notifier->start(job_id);

    //clock_t start_time = clock64();
    //while (clock64() - start_time < 10000000);

    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned grid_size = blockDim.x * gridDim.x;

    while (id < count) {
        float tmp = 1;
        for (unsigned i = 1; i <= compute_count; ++i) {
            tmp *= i;
        }
        mem[id] = tmp;
        id += grid_size;
    }

    notifier->end(job_id);
}

class DummyShortCoroutineJob : public llis::job::CoroutineJob {
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

    void one_time_init() override {
        set_num_threads_per_block(256);
        set_smem_size_per_block(0);
        set_num_registers_per_thread(32);
        set_num_blocks(5);
        unset_is_mem();

        cudaMalloc(&mem_, count_ * sizeof(*mem_));
    }

    void body(void* io_ptr) override {
        for (int i = 0; i < num_kernels; ++i) {
            if (i == num_kernels - 1) {
                set_pre_notify();
            }
            yield();
            llis::job::FinishedBlockNotifier* notifier = get_finished_block_notifier();
            dummy_kernel<<<get_num_blocks(), get_num_threads_per_block(), 0, get_cuda_stream()>>>(mem_, count_, compute_count_, get_id(), notifier);
        }
    }

  private:
    float* mem_;

    static constexpr unsigned count_ = 5000000;
    static constexpr unsigned compute_count_ = 100;
    static constexpr unsigned num_kernels = 11;
};

extern "C" {

__attribute__((visibility("default")))
llis::job::Job* init_job() {
    return new DummyShortCoroutineJob();
}

}

