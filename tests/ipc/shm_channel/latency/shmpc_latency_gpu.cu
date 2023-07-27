#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/instrument_info.h>

#include <iostream>
#include <chrono>

__global__ void kernel(float* output, float* input, unsigned count, llis::ipc::ShmPrimitiveChannelGpu<uint64_t> channel) {
//__global__ void kernel(llis::ipc::ShmPrimitiveChannelGpu<uint64_t> channel) {
    if (threadIdx.x == 0) {
        llis::job::InstrumentInfo info;
        info.is_start = true;
        info.job_id = blockIdx.x;
        channel.write(info);
    }

    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned grid_size = blockDim.x * gridDim.x;

    while (id < count) {
        output[id] += input[id];
        id += grid_size;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        llis::job::InstrumentInfo info;
        info.is_start = false;
        info.job_id = blockIdx.x;
        channel.write(info);
    }
}

int main(int argc, char** argv) {
    unsigned num_blocks = atoi(argv[1]);
    unsigned vec_len = atoi(argv[2]);
    unsigned num_iters = atoi(argv[3]);

    float* x;
    float* y;
    cudaMalloc(&x, sizeof(*x) * vec_len);
    cudaMalloc(&y, sizeof(*y) * vec_len);

    llis::ipc::ShmPrimitiveChannelGpu<uint64_t> channel("", 1024000);

    for (int i = 0; i < num_iters; ++i) {
        kernel<<<num_blocks, 100>>>(y, x, vec_len, channel.fork());
        //kernel<<<num_blocks, 100>>>(channel.fork());
        for (int j = 0; j < num_blocks * 2; ++j) {
            (void)channel.read<llis::job::InstrumentInfo>();
        }
    }

    auto start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < num_iters; ++i) {
        kernel<<<num_blocks, 100>>>(y, x, vec_len, channel.fork());
        //kernel<<<num_blocks, 100>>>(channel.fork());
        for (int j = 0; j < num_blocks * 2; ++j) {
            (void)channel.read<llis::job::InstrumentInfo>();
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    std::cout << "Time elasped: " << std::chrono::duration<double, std::micro>(end_time - start_time).count() / (double)num_iters << " us" << std::endl;
}

