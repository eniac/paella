#pragma once

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/job.h>
#include <llis/job/instrument_info.h>
#include <llis/server/server.h>

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
    struct SmAvail {
        int nregs = 0;
        int smem = 0;
        int nthrs = 0;
        int nblocks = 0;

        bool is_ok() const {
            return nregs >= 0 & smem >= 0 & nthrs >= 0 & nblocks >= 0;
        }

        void add(job::Job* job, int num) {
            // TODO: handle allocation granularity
            nregs += job->get_cur_num_registers_per_thread() * job->get_cur_num_threads_per_block() * num;
            nthrs += job->get_cur_num_threads_per_block() * num;
            smem += job->get_cur_smem_size_per_block() * num;
            nblocks += job->get_cur_num_blocks() * num;
        }

        void minus(job::Job* job, int num) {
            // TODO: handle allocation granularity
            nregs -= job->get_cur_num_registers_per_thread() * job->get_cur_num_threads_per_block() * num;
            nthrs -= job->get_cur_num_threads_per_block() * num;
            smem -= job->get_cur_smem_size_per_block() * num;
            nblocks -= job->get_cur_num_blocks() * num;
        }

        double dot(job::Job* job) const {
            double res = (double)nregs * (double)job->get_cur_num_registers_per_thread() * (double)job->get_cur_num_threads_per_block();
            res += (double)nthrs * (double)job->get_cur_num_threads_per_block();
            res += (double)smem * (double)job->get_smem_size_per_block();
            return res;
        }
    };

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
    bool job_fits(job::Job* job);
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

    std::vector<SmAvail> sm_avails_;
    SmAvail total_sm_avail_;
    std::vector<unsigned> gpc_num_blocks_;
    std::vector<unsigned> gpc_next_sms_;
    constexpr static unsigned gpc_sms_[5][8] = {{0, 10, 20, 30, 1, 11, 21, 31}, {2, 12, 22, 32, 3, 13, 23, 33}, {4, 14, 24, 34, 5, 15, 25, 35}, {6, 16, 26, 36, 7, 17, 27, 37}, {8, 18, 28, 38, 9, 19, 29, 39}};
    constexpr static unsigned total_nregs_ = 65536 * 40;
    constexpr static unsigned total_nthrs_ = 2048 * 40;
    constexpr static unsigned total_smem_ = 65536 * 40;
    constexpr static unsigned total_nblocks_ = 32 * 40;

    unsigned num_jobs_ = 0;

#ifdef PRINT_NUM_RUNNING_KERNELS
    unsigned num_running_kernels_ = 0;
#endif

    float new_job_deficit_ = 0;

    double max_resources_dot_prod_;

    JobLess job_less_;

    bool is_all_sms_full_;
};

}
}

