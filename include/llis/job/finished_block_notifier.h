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
            start_end_time_.data[0] = clock_val >> 8;
            start_end_time_.data[1] = (clock_val & 0xFF) << 8;
#endif

#ifdef LLIS_FINISHED_BLOCK_NOTIFICATION_AGG
            unsigned num_blocks = gridDim.x * gridDim.y * gridDim.z;
            unsigned total_num = atomicInc(&agg_counter_start_, num_blocks - 1) + 1;
            unsigned batch_num = total_num % noti_batch_size_;
            if (batch_num == 0) {
                InstrumentInfo info;
                info.is_start = 1;
                info.job_id = job_id;
                info.num = noti_batch_size_;
                gpu2sched_channel_.write(info);
            } else if (total_num == num_blocks) {
                InstrumentInfo info;
                info.is_start = 1;
                info.job_id = job_id;
                info.num = batch_num;
                gpu2sched_channel_.write(info);
            }
#else
            InstrumentInfo info;
            info.is_start = 1;
            info.job_id = job_id;

            gpu2sched_channel_.write(info);
#endif
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

#ifdef LLIS_FINISHED_BLOCK_NOTIFICATION_AGG
            unsigned num_blocks = gridDim.x * gridDim.y * gridDim.z;
            unsigned total_num = atomicInc(&agg_counter_end_, num_blocks - 1) + 1;
            unsigned batch_num = total_num % noti_batch_size_;
            if (batch_num == 0) {
                InstrumentInfo info;
                info.is_start = 0;
                info.job_id = job_id;
                info.num = noti_batch_size_;
                gpu2sched_channel_.write(info);
            } else if (total_num == num_blocks) {
                InstrumentInfo info;
                info.is_start = 0;
                info.job_id = job_id;
                info.num = batch_num;
                gpu2sched_channel_.write(info);
            }
#else
            InstrumentInfo info;
            info.is_start = 0;
            info.job_id = job_id;

            gpu2sched_channel_.write(info);
#endif

#ifdef LLIS_MEASURE_BLOCK_TIME
            gpu2sched_block_time_channel_.write(start_end_time_);
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
    ipc::Gpu2SchedChannel gpu2sched_channel_;

#ifdef LLIS_MEASURE_BLOCK_TIME
    BlockStartEndTime start_end_time_;
    ipc::Gpu2SchedChannel gpu2sched_block_time_channel_;
#endif

#ifdef LLIS_FINISHED_BLOCK_NOTIFICATION_AGG
    unsigned agg_counter_start_ = 0;
    unsigned agg_counter_end_ = 0;

    static constexpr unsigned noti_batch_size_ = 16;
#endif
};

}
}


