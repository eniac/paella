#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/coroutine_job.h>
#include <llis/job/context.h>
#include <llis/job/instrument.h>

#include <cstdio>

__global__ void vec_add(float* output, float* input, size_t count, unsigned long long dummy[10], llis::JobId job_id, llis::ipc::Gpu2SchedChannel gpu2sched_channel
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

    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned grid_size = blockDim.x * gridDim.x;

    while (id < count) {
        output[id] += input[id];
        id += grid_size;
    }

#ifdef LLIS_MEASURE_BLOCK_TIME
    llis::job::kernel_end(job_id, &gpu2sched_channel, &gpu2sched_block_time_channel, &start_end_time);
#else
    llis::job::kernel_end(job_id, &gpu2sched_channel);
#endif
}

class VecAddCoroutineJob : public llis::job::CoroutineJob {
  private:
    //static constexpr unsigned count_ = 128000;
    static constexpr unsigned count_ = 32000;
    static constexpr unsigned num_blocks = 200;
    //static constexpr unsigned num_blocks = 1000;
    //static constexpr unsigned num_threads_per_block = count_ / num_blocks;
    static constexpr unsigned num_threads_per_block = 10;
    //static constexpr unsigned num_iters = 55;
    static constexpr unsigned num_iters = 10;

    static constexpr unsigned num_blocks_list[] = {196,
        224,
        98,
        392,
        2688,
        28,
        168,
        504,
        28,
        168,
        2016,
        4,
        168,
        192,
        7,
        168,
        192,
        7,
        168,
        224,
        4,
        32,
        384,
        7,
        32,
        384,
        7,
        32,
        384,
        7,
        32,
        384,
        7,
        112,
        1008,
        12,
        112,
        1008,
        12,
        112,
        168,
        8,
        12,
        48,
        8,
        12,
        48,
        8,
        12,
        48,
        5,
        20,
        160,
        2,
        1000};

  public:
    size_t get_input_size() override {
        return count_ * sizeof(float);
    }

    size_t get_output_size() override {
        return count_ * sizeof(float);
    }

    size_t get_param_size() override {
        return 4;
    }

    void one_time_init() override {
        set_num_threads_per_block(num_threads_per_block);
        set_num_blocks(num_blocks);
        set_smem_size_per_block(0);

        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, vec_add);
        set_num_registers_per_thread(attr.numRegs);

        cudaMalloc(&input_dev_, get_input_size());
        cudaMalloc(&output_dev_, get_output_size());
    }

    void body(void* io_ptr) override {
        float* input = (float*)io_ptr;
        float* output = (float*)io_ptr + count_;

        set_is_mem();
        yield();
        cudaMemcpyAsync(input_dev_, input, get_input_size(), cudaMemcpyHostToDevice, get_cuda_stream());
        unset_is_mem();

        unsigned long long dummy[10];

        //set_pre_notify();

        for (int i = 0; i < num_iters; ++i) {
            //unsigned num_blocks = num_blocks_list[i];
            //unsigned num_threads_per_block = 100;
            //set_num_blocks(num_blocks);
            //set_num_threads_per_block(num_threads_per_block);
            yield();
            vec_add<<<num_blocks, num_threads_per_block, 0, get_cuda_stream()>>>(output, input, count_, dummy, get_id(), llis::job::Context::get_gpu2sched_channel()->fork()
#ifdef LLIS_MEASURE_BLOCK_TIME
                , llis::job::Context::get_gpu2sched_block_time_channel()->fork()
#endif
                );
        }

        set_is_mem();
        set_pre_notify();
        yield();
        cudaMemcpyAsync(output, output_dev_, get_output_size(), cudaMemcpyDeviceToHost, get_cuda_stream());
    }

  private:
    float* input_dev_;
    float* output_dev_;
};

extern "C" {

llis::job::Job* init_job() {
    return new VecAddCoroutineJob();
}

}

