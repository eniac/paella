#include "llis/ipc/shm_primitive_channel.h"
#include <llis/job/instrument.h>
#include <llis/job/job.h>
#include <llis/ipc/shm_channel.h>

#include <cmath>

namespace llis {
namespace job {

namespace {

__global__ void memset_impl(void* ptr, int val, size_t count, JobId job_id, ipc::Gpu2SchedChannel gpu2sched_channel) {
    kernel_start(job_id, &gpu2sched_channel);

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < count) {
        (reinterpret_cast<char*>(ptr))[id] = val;
    }

    kernel_end(job_id, &gpu2sched_channel);
}

}

void memset_res(size_t count, Job* job) {
    constexpr int num_threads_per_block = 256;

    job->set_num_blocks(std::ceil((float)count / (float)num_threads_per_block));
    job->set_num_threads_per_block(num_threads_per_block);
    job->set_num_registers_per_thread(32);
    job->set_smem_size_per_block(0);
}

void memset(void* ptr, int val, size_t count, Job* job, ipc::Gpu2SchedChannel* gpu2sched_channel) {
    constexpr int num_threads_per_block = 256;

    memset_impl<<<job->get_num_blocks(), num_threads_per_block, job->get_smem_size_per_block(), job->get_cuda_stream()>>>(ptr, val, count, job->get_id(), gpu2sched_channel->fork());
}

}
}
