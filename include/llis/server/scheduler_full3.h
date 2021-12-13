#pragma once

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/job.h>
#include <llis/job/instrument_info.h>
#include <llis/server/server.h>
#include <llis/server/gpu_resources.h>
#include <llis/utils/logging.hh>
#include <llis/job/finished_block_notifier.h>

#include <cuda_runtime.h>

#include <vector>
#include <memory>
#include <map>

#define GPU2SCHED_CHAN_SIZE 1024000
#define GPU2SCHED_CHAN_SIZE_TIME 10240000

namespace llis {
namespace server {

class JobQueue {
  public:
    void push(job::Job* job) {
        Entry entry;
        entry.job = job;

        JobMap::iterator it_priority = map_priority_.emplace(job->get_priority(), entry);
        JobMap::iterator it_fairness = map_fairness_.emplace(job->get_deficit_counter(), entry);

        it_priority->second.other_it = it_fairness;
        it_fairness->second.other_it = it_priority;
    }

    job::Job* top(double unfairness_threshold) {
        if (map_fairness_.begin()->first >= unfairness_threshold) {
            return map_fairness_.begin()->second.job;
        } else {
            return map_priority_.begin()->second.job;
        }
    }

    job::Job* pop(double unfairness_threshold) {
        if (map_fairness_.begin()->first >= unfairness_threshold) {
            Entry entry = map_fairness_.begin()->second;
            map_fairness_.erase(map_fairness_.begin());
            map_priority_.erase(entry.other_it);
            return entry.job;
        } else {
            Entry entry = map_priority_.begin()->second;
            map_priority_.erase(map_priority_.begin());
            map_fairness_.erase(entry.other_it);
            return entry.job;
        }
    }

    void clear() {
        map_priority_.clear();
        map_fairness_.clear();
    }

    bool empty() {
        // Both maps should have the same size
        return map_priority_.empty();
    }

    bool size() {
        // Both maps should have the same size
        return map_priority_.size();
    }

  private:
    struct Entry;

    using JobMap = std::multimap<double, Entry, std::greater<double>>;

    struct Entry {
        job::Job* job;
        JobMap::iterator other_it;
    };

    JobMap map_priority_;
    JobMap map_fairness_;
};

class SchedulerFull3 {
  public:
    SchedulerFull3(float unfairness_threshold, float eta);

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
    void update_deficit_counters(job::Job* job_scheduled);
    double calculate_priority(job::Job* job) const;
    double calculate_packing(job::Job* job) const;
    static float normalize_resources(job::Job* job);

    static void mem_notification_callback(void* job);

    Server* server_;
    Profiler* profiler_;
    ipc::ShmPrimitiveChannelGpu<uint64_t> gpu2sched_channel_;
#ifdef LLIS_MEASURE_BLOCK_TIME
    ipc::ShmPrimitiveChannelGpu<uint64_t> gpu2sched_block_time_channel_;
#endif
    ipc::ShmChannelCpuReader mem2sched_channel_;
    
    std::vector<cudaStream_t> cuda_streams_;
    job::FinishedBlockNotifier* finished_block_notifiers_raw_;
    std::vector<job::FinishedBlockNotifier*> finished_block_notifiers_;

    float unfairness_threshold_;
    float eta_;
    JobQueue job_queue_;

    std::vector<std::unique_ptr<job::Job>> job_id_to_job_map_;
    std::vector<JobId> unused_job_id_;

    SmResources gpu_resources_;

    unsigned num_jobs_ = 0;

#ifdef PRINT_NUM_RUNNING_KERNELS
    unsigned num_running_kernels_ = 0;
    unsigned num_running_mems_ = 0;
#endif

    unsigned num_outstanding_kernels_ = 0;
    static constexpr unsigned max_num_outstanding_kernels_ = 2;
    
    float new_job_deficit_ = 0;
};

}
}

