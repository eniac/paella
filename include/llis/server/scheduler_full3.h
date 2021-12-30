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
#include <queue>
#include <cfloat>

#define GPU2SCHED_CHAN_SIZE 1024000
#define GPU2SCHED_CHAN_SIZE_TIME 10240000

namespace llis {
namespace server {

class JobQueue {
  public:
    JobQueue(double unfairness_threshold) : unfairness_threshold_(unfairness_threshold) {}

    void push(job::Job* job) {
        JobRefId type_id = job->get_registered_job_id();

        Entry entry;
        entry.job = job;

        JobMap::iterator it_all = all_map_.emplace(job->get_priority(), entry);

        // Handle new job types
        if (type_id >= per_type_maps_.size()) { // Either create a new entry
            size_t original_size = per_type_maps_.size();
            per_type_maps_.resize(type_id + 1);

            for (size_t i = original_size; i < type_id; ++i) {
                per_type_maps_[i].deficit_counter = DBL_MAX; // DBL_MAX means this is just a placeholder
            }

            per_type_maps_[type_id].deficit_counter = new_type_deficit_;

            ++num_types_;
        } else if (per_type_maps_[type_id].deficit_counter == DBL_MAX) { // Or (re)use an existing but unused entry
            per_type_maps_[type_id].deficit_counter = new_type_deficit_;

            ++num_types_;
        }

        TypeEntry* type_entry = &per_type_maps_[type_id];

        // Put a type (back) into the type map when there was not jobs of that type but the current job is of that type
        if (type_entry->job_map->size() == 0) {
            type_entry->it = type_map_.emplace(type_entry->deficit_counter, type_entry->job_map.get());
        }

        // Put the job into the corresponding per type map

        JobMap::iterator it_per_type = type_entry->job_map->emplace(job->get_priority(), entry);

        it_all->second.other_it = it_per_type;
        it_per_type->second.other_it = it_all;
    }

    job::Job* top() {
        // Can assume that type_map_ is always non-empty
        // Also, all per type maps in type_map_ are non-empty
        if (type_map_.begin()->first >= unfairness_threshold_) {
            //printf("Fairness triggered\n");
            JobMap* per_type_job_map = type_map_.begin()->second;
            return per_type_job_map->begin()->second.job;
        } else {
            return all_map_.begin()->second.job;
        }
    }

    job::Job* pop() {
        Entry entry;

        // Can assume that type_map_ is always non-empty
        // Also, all per type maps in type_map_ are non-empty
        if (type_map_.begin()->first >= unfairness_threshold_) {
            JobMap* per_type_job_map = type_map_.begin()->second;

            entry = per_type_job_map->begin()->second;
            per_type_job_map->erase(per_type_job_map->begin());

            all_map_.erase(entry.other_it);
        } else {
            entry = all_map_.begin()->second;
            all_map_.erase(all_map_.begin());

            JobMap* per_type_job_map = per_type_maps_[entry.job->get_registered_job_id()].job_map.get();
            per_type_job_map->erase(entry.other_it);
        }

        JobRefId type_id = entry.job->get_registered_job_id();
        TypeEntry* type_entry = &per_type_maps_[type_id];
        TypeMap::iterator type_it = type_entry->it;

        if (type_entry->job_map->empty()) {
            // Update the deficit counter in per_type_maps. It is never updated until it is removed from type_map_
            type_entry->deficit_counter = type_it->first - 1.;
            type_map_.erase(type_it);
        } else {
            auto type_node = type_map_.extract(type_it);
            type_node.key() -= 1.;
            type_entry->it = type_map_.insert(std::move(type_node));
        }

        const double fair_share = 1. / (double)num_types_;
        new_type_deficit_ -= fair_share;
        unfairness_threshold_ -= fair_share;

        // To avoid overflow
        if (new_type_deficit_ < 10. - DBL_MAX) {
            printf("Warning: deficit counter almost overflow\n");
            rebuild_type_map();
            unfairness_threshold_ -= new_type_deficit_;
            new_type_deficit_ = 0;
        }

        return entry.job;
    }

    bool empty() const {
        return all_map_.empty();
    }

    size_t size() const {
        return all_map_.size();
    }

  private:
    void rebuild_type_map() {
        for (auto& p : per_type_maps_) {
            if (p.deficit_counter == DBL_MAX) {
                // DBL_MAX means this is only a placeholder
                continue;
            }

            if (p.job_map->empty()) {
                // When empty, the job map is not included in the type map.
                // So, we update the counter in per_type_maps_
                // Also, we should not insert it into the type map.
                p.deficit_counter -= new_type_deficit_;
            } else {
                // When non-empty, it is in the type map.
                // So, we update the counter in the map. The counter in per_type_maps_ is useless until it is removed from type map. When that happens (in pop()), it will be updated
                auto type_node = type_map_.extract(p.it);
                // TODO: check this logic. This may cause overflow
                type_node.key() -= new_type_deficit_;
                p.it = type_map_.insert(std::move(type_node));
            }
        }
    }

    struct Entry;

    using JobMap = std::multimap<double, Entry, std::greater<double>>;
    using TypeMap = std::multimap<double, JobMap*, std::greater<double>>;

    struct Entry {
        job::Job* job;
        JobMap::iterator other_it;
    };

    struct TypeEntry {
        double deficit_counter;
        std::unique_ptr<JobMap> job_map;
        TypeMap::iterator it;

        TypeEntry() : job_map(std::make_unique<JobMap>()) {}
    };

    double unfairness_threshold_;
    double new_type_deficit_ = 0;
    unsigned num_types_; // FIXME: handle unregister of a type

    // Contains all jobs, sorted descendingly in priority
    JobMap all_map_;

    // A list of job maps. The i-th map contains all jobs of type i, sorted descending in priority.
    // FIXME: handle when unregister when there are still running jobs of that type
    std::vector<TypeEntry> per_type_maps_;

    // A map of per type JobMap. Sorted descendingly in deficit counter
    TypeMap type_map_;
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
};

}
}

