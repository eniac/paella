#pragma once

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/job.h>
#include <llis/job/instrument_info.h>
#include <llis/server/server.h>
#include <llis/server/gpu_resources.h>

#include <cuda_runtime.h>

#include <deque>
#include <queue>
#include <vector>
#include <memory>

namespace llis {
namespace server {

class Scheduler {
  public:
    Scheduler(float unfairness_threshold, float eta);

    void set_server(Server* server);

    void handle_new_job(std::unique_ptr<job::Job> job);
    void try_handle_block_start_finish();

  private:
    class JobLess {
      public:
        JobLess(float unfairness_threshold) : unfairness_threshold_(unfairness_threshold) {}

        bool operator()(const job::Job* left, const job::Job* right) const {
            int is_left_unfair = left->get_deficit_counter() >= unfairness_threshold_;
            int is_right_unfair = right->get_deficit_counter() >= unfairness_threshold_;

            if (is_left_unfair < is_right_unfair) {
                return true;
            } else if (is_left_unfair == is_right_unfair) {
                return left->get_priority() < right->get_priority();
            } else {
                return false;
            }
        }

      private:
        float unfairness_threshold_;
    };

    void handle_block_start_finish();
    void handle_block_start(const job::InstrumentInfo& info);
    void handle_block_finish(const job::InstrumentInfo& info);
    void handle_mem_finish();

    void schedule_job();
    void choose_sms(job::Job* job);
    void update_deficit_counters(job::Job* job_scheduled);

    double calculate_priority(job::Job* job) const;
    double calculate_packing(job::Job* job) const;
    static float normalize_resources(job::Job* job);

    Server* server_;
    ipc::ShmPrimitiveChannelGpu<uint64_t> gpu2sched_channel_;
    ipc::ShmChannel mem2sched_channel_;
    
    std::vector<cudaStream_t> cuda_streams_;

    float unfairness_threshold_;
    float eta_;
    std::vector<job::Job*> job_queue_;

    std::vector<std::unique_ptr<job::Job>> job_id_to_job_map_;
    std::vector<JobId> unused_job_id_;

    GpuResources gpu_resources_;

    unsigned num_jobs_ = 0;

#ifdef PRINT_NUM_RUNNING_KERNELS
    unsigned num_running_kernels_ = 0;
#endif

    float new_job_deficit_ = 0;

    JobLess job_less_;
};

}
}

