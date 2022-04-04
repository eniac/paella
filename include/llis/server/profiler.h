#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/ipc/defs.h>
#include <llis/job/job.h>

#include <chrono>
#include <vector>

namespace llis {
namespace server {

class Profiler {
  public:
    enum class JobEvent {
        JOB_SUBMITTED,
        KERNEL_SCHED_START,
        KERNEL_SCHED_ABORT,
        KERNEL_SUBMIT_START,
        KERNEL_SUBMIT_END,
        KERNEL_FINISHED,
        JOB_FINISHED,
    };

    enum class ResourceEvent {
        ACQUIRE,
        RELEASE
    };

    Profiler(ipc::ShmChannelCpuReader* c2s_channel) : c2s_channel_(c2s_channel) {}

    void handle_cmd();
    void handle_cmd_save();

    void save(const std::string& path);

    void record_kernel_info(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time, unsigned num_blocks, unsigned num_threads_per_block, unsigned smem_size_per_block, unsigned num_registers_per_thread, double priority, unsigned job_ref_id);
    void record_block_exec_time(unsigned long long start_time, unsigned long long end_time);
    //void record_kernel_sm_exec_time(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time);

    void recrod_kernel_block_mis_alloc(unsigned total, unsigned total_wrong_prediction, unsigned total_wrong_prediction_sm);

    void record_run_next_time(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time, unsigned num_blocks);

    void record_job_event(JobId job_id, JobEvent event);

    void record_resource_event(job::Job* job, unsigned num, ResourceEvent event);

  private:
    ipc::ShmChannelCpuReader* c2s_channel_;

    bool kernel_info_flag_ = false;
    std::vector<std::tuple<std::chrono::time_point<std::chrono::steady_clock>, std::chrono::time_point<std::chrono::steady_clock>, unsigned, unsigned, unsigned, unsigned, double, unsigned>> kernel_info_;

    bool block_exec_times_flag_ = false;
    std::vector<std::pair<unsigned long long, unsigned long long>> block_exec_times_;
    //std::vector<std::pair<std::chrono::time_point<std::chrono::steady_clock>, std::chrono::time_point<std::chrono::steady_clock>>> block_exec_times_;

    bool kernel_block_mis_alloc_flag_ = false; 
    std::vector<std::tuple<unsigned, unsigned, unsigned>> kernel_block_mis_alloc_;

    bool run_next_times_flag_ = false;
    std::vector<std::tuple<std::chrono::time_point<std::chrono::steady_clock>, std::chrono::time_point<std::chrono::steady_clock>, unsigned>> run_next_times_;

    bool job_events_flag_ = false;
    std::vector<std::vector<std::pair<JobEvent, std::chrono::time_point<std::chrono::steady_clock>>>> jobs_events_cur_;
    std::vector<std::vector<std::pair<JobEvent, std::chrono::time_point<std::chrono::steady_clock>>>> jobs_events_all_;

    bool resource_events_flag_ = false;
    std::vector<std::tuple<JobId, std::string, ResourceEvent, std::chrono::time_point<std::chrono::steady_clock>, unsigned, unsigned, unsigned, unsigned>> resource_events_;
};

}
}

