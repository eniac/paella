#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/instrument.h>
#include <llis/job/job.h>
#include <llis/ipc/shm_channel.h>

#include <cmath>

namespace llis {
namespace job {

namespace {

__global__ void memset_impl(void* ptr, int val, size_t count, JobId job_id, ipc::Gpu2SchedChannel gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
        , ipc::Gpu2SchedChannel gpu2sched_block_time_channel
#endif
) {
#ifdef LLIS_MEASURE_BLOCK_TIME
    BlockStartEndTime start_end_time;
    kernel_start(job_id, &gpu2sched_channel, &start_end_time);
#else
    kernel_start(job_id, &gpu2sched_channel);
#endif

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < count) {
        (reinterpret_cast<char*>(ptr))[id] = val;
    }

#ifdef LLIS_MEASURE_BLOCK_TIME
    kernel_end(job_id, &gpu2sched_channel, &gpu2sched_block_time_channel, &start_end_time);
#else
    kernel_end(job_id, &gpu2sched_channel);
#endif
}

}

void memset_res(size_t count, Job* job) {
    constexpr int num_threads_per_block = 256;

    job->set_num_blocks(std::ceil((float)count / (float)num_threads_per_block));
    job->set_num_threads_per_block(num_threads_per_block);
    job->set_num_registers_per_thread(32);
    job->set_smem_size_per_block(0);
}

void memset(void* ptr, int val, size_t count, Job* job, ipc::Gpu2SchedChannel* gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
        , ipc::Gpu2SchedChannel* gpu2sched_block_time_channel
#endif
) {
    constexpr int num_threads_per_block = 256;

#ifdef LLIS_MEASURE_BLOCK_TIME
    memset_impl<<<job->get_num_blocks(), num_threads_per_block, job->get_smem_size_per_block(), job->get_cuda_stream()>>>(ptr, val, count, job->get_id(), gpu2sched_channel->fork(), gpu2sched_block_time_channel->fork());
#else
    memset_impl<<<job->get_num_blocks(), num_threads_per_block, job->get_smem_size_per_block(), job->get_cuda_stream()>>>(ptr, val, count, job->get_id(), gpu2sched_channel->fork());
#endif
}

}
}
