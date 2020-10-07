#pragma once

#include <llis/ipc/shm_channel.h>

namespace llis {
namespace job {

__device__ void kernel_start(void* job, ipc::ShmChannelGpu* gpu2sched_channel) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));

        gpu2sched_channel->acquire_writer_lock();
        gpu2sched_channel->write(true);
        gpu2sched_channel->write(job);
        gpu2sched_channel->write(smid);
        gpu2sched_channel->release_writer_lock();
    }
}

__device__ void kernel_end(void* job, ipc::ShmChannelGpu* gpu2sched_channel) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        gpu2sched_channel->acquire_writer_lock();
        gpu2sched_channel->write(false);
        gpu2sched_channel->write(job);
        gpu2sched_channel->release_writer_lock();
    }
}

}
}

