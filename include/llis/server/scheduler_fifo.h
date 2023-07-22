#pragma once

#include <llis/job/finished_block_notifier.h>
#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/job.h>
#include <llis/job/instrument_info.h>
#include <llis/server/server.h>
#include <llis/server/gpu_resources.h>
#include <llis/utils/logging.hh>
#include <llis/server/scheduler.h>

#include <cuda_runtime.h>

#include <deque>
#include <queue>
#include <vector>
#include <memory>

#define GPU2SCHED_CHAN_SIZE 1024000
#define GPU2SCHED_CHAN_SIZE_TIME 10240000

namespace llis {
namespace server {

class SchedulerFifo : public Scheduler {
  public:
    SchedulerFifo(unsigned num_streams, unsigned sched_sleep);

    void handle_new_job(std::unique_ptr<job::Job> job) override;
    void try_handle_block_start_finish() override;

  private:
    void handle_block_start_finish();
#ifdef LLIS_MEASURE_BLOCK_TIME
    void handle_block_start_end_time();
#endif
    void handle_block_start(const job::InstrumentInfo& info);
    void handle_block_finish(const job::InstrumentInfo& info);
    void handle_mem_finish();

    void schedule_job();

    static void mem_notification_callback(void* job);

    ipc::ShmPrimitiveChannelGpu<uint64_t> gpu2sched_channel_;
#ifdef LLIS_MEASURE_BLOCK_TIME
    ipc::ShmPrimitiveChannelGpu<uint64_t> gpu2sched_block_time_channel_;
#endif
    ipc::ShmChannelCpuReader mem2sched_channel_;
    
    std::vector<cudaStream_t> cuda_streams_;
    job::FinishedBlockNotifier* finished_block_notifiers_raw_;
    std::vector<job::FinishedBlockNotifier*> finished_block_notifiers_;

    std::queue<job::Job*> job_queue_;

    std::vector<std::unique_ptr<job::Job>> job_id_to_job_map_;
    std::vector<JobId> unused_job_id_;

    std::vector<unsigned> remaining_num_blocks_;
    std::vector<int> pre_notify_blocks_;
    std::vector<bool> pre_notify_sent_;

    unsigned num_jobs_ = 0;

#ifdef PRINT_NUM_RUNNING_JOBS
    unsigned num_running_jobs_ = 0;
#endif
#ifdef PRINT_NUM_RUNNING_BLOCKS
    unsigned num_running_blocks_ = 0;
#endif
};

}
}

