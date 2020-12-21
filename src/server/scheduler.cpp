#include "llis/ipc/shm_channel.h"
#include <llis/server/scheduler.h>

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/server/server.h>
#include <llis/job/job.h>
#include <llis/job/context.h>
#include <llis/job/instrument_info.h>

#include <sys/socket.h>
#include <sys/un.h>

#include <memory>
#include <string>
#include <algorithm>
#include <chrono>
#include <cfloat>

//#define PRINT_SCHEDULE_TIME
#define PRINT_SORT_TIME

namespace llis {
namespace server {

void mem_notification_callback(void* job);

Scheduler::Scheduler(float unfairness_threshold) : server_(nullptr), gpu2sched_channel_(1024000), mem2sched_channel_(10240), cuda_streams_(100), unfairness_threshold_(unfairness_threshold) { // TODO: size of the channel must be larger than number of total blocks * 2
    job::Context::set_gpu2sched_channel(&gpu2sched_channel_);
    job::Context::set_mem2sched_channel(&mem2sched_channel_);

    for (auto& stream : cuda_streams_) {
        cudaStreamCreate(&stream);
    }

    // TODO: query the numbers
    // Some of them can be zero because some smid may not reflect an actual SM
    sm_avails_.resize(40);
    for (auto& sm_avail : sm_avails_) {
        sm_avail.nregs = 65536;
        sm_avail.nthrs = 2048;
        sm_avail.smem = 65536;
        sm_avail.nblocks = 32;
    }

    gpc_num_blocks_.resize(5);
    gpc_next_sms_.resize(5);
}

void Scheduler::set_server(Server* server) {
    server_ = server;
}

void Scheduler::try_handle_block_start_finish() {
    if (gpu2sched_channel_.can_read<job::InstrumentInfo>()) {
        handle_block_start_finish();
    }

    if (mem2sched_channel_.can_read()) {
        handle_mem_finish();
    }
}

void Scheduler::handle_block_start_finish() {
    job::InstrumentInfo info = gpu2sched_channel_.read<job::InstrumentInfo>();
    
    if (info.is_start) {
        handle_block_start(info);
    } else {
        handle_block_finish(info);
    }
}

void Scheduler::handle_block_start(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id];

    if (!job->has_predicted_smid(info.smid)) {
        sm_avails_[info.smid].minus(job, 1);
    } else {
        job->dec_predicted_smid(info.smid);
    }

    if (!job->mark_block_start()) {
        const unsigned* predicted_smid_nums = job->get_predicted_smid_nums();
        for (unsigned smid = 0; smid < 40; ++smid) {
            unsigned num = predicted_smid_nums[smid];

            // TODO: handle allocation granularity
            sm_avails_[smid].add(job, num);
        }
    }
}

void Scheduler::handle_block_finish(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id];

    job->mark_block_finish();
    if (!job->is_running()) {
        if (!job->has_next()) {
            server_->notify_job_ends(job);
            --num_jobs_;
        }

        auto end_time = std::chrono::steady_clock::now();
        auto start_time = job->get_stage_start_time();
        double length = std::chrono::duration<double, std::micro>(end_time - start_time).count();
        server_->update_job_stage_length(job, job->get_cur_stage(), length);

#ifdef PRINT_NUM_RUNNING_KERNELS
        --num_running_kernels_;
        printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif

        cuda_streams_.push_back(job->get_cuda_stream());
    }

    sm_avails_[info.smid].add(job, 1);

    schedule_job();
}

void Scheduler::handle_mem_finish() {
    job::Job* job;
    mem2sched_channel_.read(&job);

    if (!job->has_next()) {
        server_->notify_job_ends(job);
        --num_jobs_;
    }

    job->unset_running();

    auto end_time = std::chrono::steady_clock::now();
    auto start_time = job->get_stage_start_time();
    double length = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    server_->update_job_stage_length(job, job->get_cur_stage(), length);

#ifdef PRINT_NUM_RUNNING_KERNELS
    --num_running_kernels_;
    printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif

    cuda_streams_.push_back(job->get_cuda_stream());

    schedule_job();
}

void Scheduler::handle_new_job(std::unique_ptr<job::Job> job) {
    if (unused_job_id_.empty()) {
        job->set_id(job_id_to_job_map_.size());
        job_id_to_job_map_.push_back(job.get());
    } else {
        job->set_id(unused_job_id_.back());
        unused_job_id_.pop_back();
        job_id_to_job_map_[job->get_id()] = job.get();
    }

    if (job->has_next()) {
        server_->set_job_stage_resource(job.get(), job->get_cur_stage() + 1, normalize_resources(job.get()));
        job->set_priority(calculate_priority(job.get()));
    } else {
        job->set_priority(-DBL_MAX); // push it to the end of the list so that it can be easily removed later
    }

    jobs_.push_back(std::move(job));

    ++num_jobs_;

    schedule_job();
}

void Scheduler::schedule_job() {
    // Sort the job list in descending order of priority
#ifdef PRINT_SCHEDULE_TIME
    auto start_schedule_time = std::chrono::steady_clock::now();
    static unsigned num_scheduled_stages = 0;
    static double total_schedule_time = 0;
    constexpr unsigned schedule_time_print_interval = 100000;
    static unsigned schedule_time_next_print = schedule_time_print_interval;
#endif
#ifdef PRINT_SORT_TIME
    constexpr unsigned sort_time_next_print = 100000;
    static unsigned sort_time_i = 0;
    auto start_sort_time = std::chrono::steady_clock::now();
#endif
    std::sort(jobs_.begin(), jobs_.end(), [this](const std::unique_ptr<job::Job>& left, const std::unique_ptr<job::Job>& right) {
        int is_left_unfair = left->get_deficit_counter() >= unfairness_threshold_;
        int is_right_unfair = right->get_deficit_counter() >= unfairness_threshold_;

        if (is_left_unfair > is_right_unfair) {
            return true;
        } else if (is_left_unfair == is_right_unfair) {
            return left->get_priority() > right->get_priority();
        } else {
            return false;
        }
    });
#ifdef PRINT_SORT_TIME
    if (sort_time_i >= sort_time_next_print) {
        auto end_sort_time = std::chrono::steady_clock::now();
        double time_taken_to_sort = std::chrono::duration<double, std::micro>(end_sort_time - start_sort_time).count();
        printf("Sort time: %lf\n", time_taken_to_sort);
        sort_time_i = 0;
    }
    ++sort_time_i;
#endif

    // All jobs that does not have a next stage to run are pushed to the end
    while (!jobs_.empty() && !jobs_.back()->has_next() && !jobs_.back()->is_running()) {
        std::unique_ptr<job::Job> job = std::move(jobs_.back());
        jobs_.pop_back();

        unused_job_id_.push_back(job->get_id());

        server_->release_job_instance(std::move(job));
    }

    if (cuda_streams_.empty()) {
        return;
    }

#ifdef PRINT_SCHEDULE_TIME
    bool has_scheduled = false;
#endif

    // TODO: do actual scheduling. Now it is just running whatever runnable, FIFO
    for (const auto& job : jobs_) {
        if (job->has_next() && !job->is_running()) {
#ifdef PRINT_SCHEDULE_TIME
            ++num_scheduled_stages;
            has_scheduled = true;
#endif
            if (!job->has_started()) {
                //server_->notify_job_starts(job.get());
                job->set_started();
            }

            job->set_running(cuda_streams_.back());
            cuda_streams_.pop_back();

#ifdef PRINT_NUM_RUNNING_KERNELS
            ++num_running_kernels_;
            printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif

            bool job_is_mem = job->is_mem();

            bool fits;
            if (job_is_mem) {
                fits = true;
            } else {
                fits = job_fits(job.get());
            }

            if (job->is_pre_notify()) {
                server_->notify_job_starts(job.get());
            }

            job::Context::set_current_job(job.get());
            job->run_next();
            
            // Note: after run_next, the is_mem flag may change, but we want to use the old one
            if (job_is_mem) {
                cudaLaunchHostFunc(job->get_cuda_stream(), mem_notification_callback, job.get());
            }

            if (job->has_next()) {
                server_->set_job_stage_resource(job.get(), job->get_cur_stage() + 1, normalize_resources(job.get()));
                job->set_priority(calculate_priority(job.get()));
            } else {
                job->set_priority(-DBL_MAX); // push it to the end of the list so that it can be easily removed later
            }

            if (!fits || cuda_streams_.empty()) {
                break;
            }
        }
    }

#ifdef PRINT_SCHEDULE_TIME
    if (has_scheduled) {
        auto end_schedule_time = std::chrono::steady_clock::now();
        total_schedule_time += std::chrono::duration<double, std::micro>(end_schedule_time - start_schedule_time).count();
        if (num_scheduled_stages >= schedule_time_next_print) {
            printf("Schedule time, # stages: %lf %u %lf\n", total_schedule_time, num_scheduled_stages, total_schedule_time / num_scheduled_stages);
            next_print += schedule_time_print_interval;
        }
    }
#endif
}

bool Scheduler::job_fits(job::Job* job) {
    unsigned num_avail_blocks = 0;

    for (auto& sm_avail : sm_avails_) {
        if (!sm_avail.is_ok()) {
            continue;
        }

        unsigned tmp = sm_avail.nblocks;

        if (job->get_num_threads_per_block() > 0) {
            tmp = std::min(tmp, sm_avail.nregs / (job->get_num_registers_per_thread() * job->get_num_threads_per_block()));
            tmp = std::min(tmp, sm_avail.nthrs / job->get_num_threads_per_block());
        }

        if (job->get_smem_size_per_block() > 0) {
            tmp = std::min(tmp, sm_avail.smem / job->get_smem_size_per_block());
        }

        num_avail_blocks += tmp;

        if (num_avail_blocks >= job->get_num_blocks()) {
            return true;
        }
    }

    return false;
}

void Scheduler::choose_sms(job::Job* job) {
    unsigned num_blocks = job->get_num_blocks();

    for (unsigned blockId = 0; blockId < num_blocks; ++blockId) {
        unsigned gpc = std::min_element(gpc_num_blocks_.begin(), gpc_num_blocks_.end()) - gpc_num_blocks_.begin();
        unsigned smid = gpc_sms_[gpc][gpc_next_sms_[gpc]];

        ++gpc_num_blocks_[gpc];
        gpc_next_sms_[gpc] = (gpc_next_sms_[gpc] + 1) % 8;

        sm_avails_[smid].minus(job, 1);

        // TODO: handle overusing resources

        job->add_predicted_smid(smid);
    }
}

void Scheduler::update_deficit_counters(job::Job* job_scheduled) {
    float val = 1. / num_jobs_;

    for (auto& job : jobs_) {
        if (job.get() == job_scheduled) {
            job->inc_deficit_counter(val - 1.);
        } else {
            job->inc_deficit_counter(val);
        }
    }
}

float Scheduler::normalize_resources(job::Job* job) {
    return ((float)(job->get_num_registers_per_thread() * job->get_num_threads_per_block()) / total_nregs_ + (float)job->get_num_threads_per_block() / total_nthrs_ + (float)job->get_smem_size_per_block() / total_smem_) * job->get_num_blocks() + (float)job->get_num_blocks() / total_nblocks_;
}

double Scheduler::calculate_priority(job::Job* job) const {
    return -server_->get_job_remaining_rl(job, job->get_cur_stage() + 1);
}

void mem_notification_callback(void* job) {
    ipc::ShmChannel* channel = job::Context::get_mem2sched_channel();
    channel->write(job);
}

}
}

