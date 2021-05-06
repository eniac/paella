#pragma once

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/job.h>
#include <llis/job/instrument_info.h>
#include <llis/server/server.h>
#include <llis/server/gpu_resources.h>
#include <llis/utils/logging.hh>

#include <cuda_runtime.h>

#include <deque>
#include <queue>
#include <vector>
#include <memory>

#define GPU2SCHED_CHAN_SIZE 1024000
#define GPU2SCHED_CHAN_SIZE_TIME 10240000

namespace llis {
namespace server {

class SchedulerFifo2 {
  public:
    SchedulerFifo2(float unfairness_threshold, float eta);

    void set_server(Server* server);

    void handle_new_job(std::unique_ptr<job::Job> job);
    void try_handle_block_start_finish();

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

    Server* server_;
    Profiler* profiler_;
    ipc::ShmPrimitiveChannelGpu<uint64_t> gpu2sched_channel_;
#ifdef LLIS_MEASURE_BLOCK_TIME
    ipc::ShmPrimitiveChannelGpu<uint64_t> gpu2sched_block_time_channel_;
#endif
    ipc::ShmChannel mem2sched_channel_;
    
    std::vector<cudaStream_t> cuda_streams_;

    std::queue<job::Job*> job_queue_all_;
    std::queue<job::Job*> job_queue_;

    std::vector<std::unique_ptr<job::Job>> job_id_to_job_map_;
    std::vector<JobId> unused_job_id_;

    unsigned num_jobs_ = 0;

    unsigned num_started_jobs_ = 0;
    static constexpr unsigned max_num_started_jobs_ = 5;

#ifdef PRINT_NUM_RUNNING_KERNELS
    unsigned num_running_kernels_ = 0;
    unsigned num_running_mems_ = 0;
#endif

    unsigned num_outstanding_kernels_ = 0;
    static constexpr unsigned max_num_outstanding_kernels_ = 1;
};

}
}

