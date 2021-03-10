#pragma once

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/instrument_info.h>

namespace llis {
namespace job {

__device__ inline void kernel_start(JobId job_id, ipc::Gpu2SchedChannel* gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
        , BlockStartEndTime* start_end_time
#endif
) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
#ifdef LLIS_MEASURE_BLOCK_TIME
        unsigned clock_val = clock64() >> 8;
        clock_val &= 0xFFFFFF;
        start_end_time->data[0] = clock_val >> 8;
        start_end_time->data[1] = (clock_val & 0xFF) << 8;
#endif

        unsigned smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));

        InstrumentInfo info;
        info.is_start = 1;
        info.smid = smid;
        info.job_id = job_id;

        gpu2sched_channel->write(info);
    }
}

__device__ inline void kernel_end(JobId job_id, ipc::Gpu2SchedChannel* gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
        , ipc::Gpu2SchedChannel* gpu2sched_block_time_channel
        , BlockStartEndTime* start_end_time
#endif
) {
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
#ifdef LLIS_MEASURE_BLOCK_TIME
        unsigned clock_val = clock64() >> 8;
        clock_val &= 0xFFFFFF;
        start_end_time->data[1] |= clock_val >> 16;
        start_end_time->data[2] = clock_val & 0xFFFF;
#endif

        unsigned smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));

        InstrumentInfo info;
        info.is_start = 0;
        info.smid = smid;
        info.job_id = job_id;

        gpu2sched_channel->write(info);
#ifdef LLIS_MEASURE_BLOCK_TIME
        gpu2sched_block_time_channel->write(*start_end_time);
#endif
    }
}

}
}

