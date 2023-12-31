#pragma once

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/job.h>
#include <llis/job/instrument_info.h>
#include <llis/server/server.h>
#include <llis/server/gpu_resources.h>
#include <llis/utils/logging.hh>
#include <llis/job/finished_block_notifier.h>
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

class SchedulerFifo2 : public Scheduler {
  public:
    SchedulerFifo2(unsigned num_streams, unsigned sched_sleep);

    void handle_new_job(std::unique_ptr<job::Job> job) override;
    void try_handle_block_start_finish() override;

  private:
    class JobCompare {
      public:
        bool operator() (const job::Job* left, const job::Job* right) const {
            return left->get_unique_id() > right->get_unique_id();
        }
    };

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

    std::priority_queue<job::Job*, std::vector<job::Job*>, JobCompare> job_queue_;

    std::vector<std::unique_ptr<job::Job>> job_id_to_job_map_;
    std::vector<JobId> unused_job_id_;

    unsigned num_jobs_ = 0;

#ifdef PRINT_NUM_RUNNING_KERNELS
    unsigned num_running_kernels_ = 0;
    unsigned num_running_mems_ = 0;
#endif
};

}
}

