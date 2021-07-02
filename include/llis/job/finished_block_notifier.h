#pragma once

#include <llis/job/instrument_info.h>
#include <llis/ipc/shm_primitive_channel.h>

namespace llis {
namespace job {

class FinishedBlockNotifier {
  public:
    FinishedBlockNotifier(ipc::Gpu2SchedChannel* gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
            , ipc::Gpu2SchedChannel* gpu2sched_block_time_channel
#endif
        );

#ifdef __CUDACC__
    __device__ __inline__ void start(JobId job_id) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
#ifdef LLIS_MEASURE_BLOCK_TIME
            unsigned clock_val = clock64() >> 8;
            clock_val &= 0xFFFFFF;
            start_end_time_->data[0] = clock_val >> 8;
            start_end_time_->data[1] = (clock_val & 0xFF) << 8;
#endif

            unsigned smid;
            asm("mov.u32 %0, %smid;" : "=r"(smid));

            InstrumentInfo info;
            info.is_start = 1;
            info.smid = smid;
            info.job_id = job_id;

            gpu2sched_channel_.write(info);
        }
    }

    __device__ __inline__ void end(JobId job_id) {
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
#ifdef LLIS_MEASURE_BLOCK_TIME
            unsigned clock_val = clock64() >> 8;
            clock_val &= 0xFFFFFF;
            start_end_time_.data[1] |= clock_val >> 16;
            start_end_time_.data[2] = clock_val & 0xFFFF;
#endif

            unsigned smid;
            asm("mov.u32 %0, %smid;" : "=r"(smid));

            InstrumentInfo info;
            info.is_start = 0;
            info.smid = smid;
            info.job_id = job_id;

            gpu2sched_channel_.write(info);
#ifdef LLIS_MEASURE_BLOCK_TIME
            gpu2sched_block_time_channel_.write(*start_end_time);
#endif
        }
    }
#endif

    static FinishedBlockNotifier* create_array(unsigned num, ipc::Gpu2SchedChannel* gpu2sched_channel
#ifdef LLIS_MEASURE_BLOCK_TIME
            , ipc::Gpu2SchedChannel* gpu2sched_block_time_channel
#endif
        );

    static void free_array(FinishedBlockNotifier* ptr);

  private:
    unsigned counter_ = 0;
    ipc::Gpu2SchedChannel gpu2sched_channel_;
#ifdef LLIS_MEASURE_BLOCK_TIME
    BlockStartEndTime start_end_time_;
    ipc::Gpu2SchedChannel gpu2sched_block_time_channel_;
#endif
};

}
}


