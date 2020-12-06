#pragma once

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/instrument_info.h>

namespace llis {
namespace job {

__device__ inline void kernel_start(JobId job_id, ipc::Gpu2SchedChannel* gpu2sched_channel) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));

        InstrumentInfo info;
        info.is_start = 1;
        info.smid = smid;
        info.job_id = job_id;

        gpu2sched_channel->write(info);
    }
}

__device__ inline void kernel_end(JobId job_id, ipc::Gpu2SchedChannel* gpu2sched_channel) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        InstrumentInfo info;
        info.is_start = 0;
        info.job_id = job_id;

        gpu2sched_channel->write(info);
    }
}

}
}

