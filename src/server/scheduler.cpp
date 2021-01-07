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

namespace llis {
namespace server {

void mem_notification_callback(void* job);

Scheduler::Scheduler(float unfairness_threshold, float eta) : server_(nullptr), gpu2sched_channel_(1024000), mem2sched_channel_(10240), cuda_streams_(500), unfairness_threshold_(unfairness_threshold), eta_(eta), job_queue_(JobLess(unfairness_threshold_)) { // TODO: size of the channel must be larger than number of total blocks * 2
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

        total_sm_avail_.nregs += 65536;
        total_sm_avail_.nthrs += 2048;
        total_sm_avail_.smem += 65536;
        total_sm_avail_.nblocks += 32;
    }

    gpc_num_blocks_.resize(5);
    gpc_next_sms_.resize(5);

    max_resources_dot_prod_ = (double)total_sm_avail_.nregs * (double)total_sm_avail_.nregs;
    max_resources_dot_prod_ += (double)total_sm_avail_.nthrs * (double)total_sm_avail_.nthrs;
    max_resources_dot_prod_ += (double)total_sm_avail_.smem * (double)total_sm_avail_.smem;
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
    job::Job* job = job_id_to_job_map_[info.job_id].get();

    if (!job->has_predicted_smid(info.smid)) {
        sm_avails_[info.smid].minus(job, 1);
        total_sm_avail_.minus(job, 1);
    } else {
        job->dec_predicted_smid(info.smid);
    }

    if (!job->mark_block_start()) {
        const unsigned* predicted_smid_nums = job->get_predicted_smid_nums();
        for (unsigned smid = 0; smid < 40; ++smid) {
            unsigned num = predicted_smid_nums[smid];

            // TODO: handle allocation granularity
            sm_avails_[smid].add(job, num);
            total_sm_avail_.add(job, num);
        }
    }
}

void Scheduler::handle_block_finish(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id].get();

    job->mark_block_finish();
    if (!job->is_running()) {
        if (job->has_next()) {
            job_queue_.push(job);
        } else {
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
    total_sm_avail_.add(job, 1);

    if (!job->is_running() && !job->has_next()) {
        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_id_to_job_map_[info.job_id]));
    }

    schedule_job();
}

void Scheduler::handle_mem_finish() {
    job::Job* job;
    mem2sched_channel_.read(&job);

    if (job->has_next()) {
        job_queue_.push(job);
    } else {
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

    if (!job->has_next()) {
        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_id_to_job_map_[job->get_id()]));
    }

    schedule_job();
}

void Scheduler::handle_new_job(std::unique_ptr<job::Job> job_) {
    job::Job* job = job_.get();
    if (!job->has_next()) {
        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_));
        return;
    }

    if (unused_job_id_.empty()) {
        job->set_id(job_id_to_job_map_.size());
        job_id_to_job_map_.push_back(std::move(job_));
    } else {
        job->set_id(unused_job_id_.back());
        unused_job_id_.pop_back();
        job_id_to_job_map_[job->get_id()] = std::move(job_);
    }

    server_->set_job_stage_resource(job, job->get_cur_stage() + 1, normalize_resources(job));
    job->set_priority(calculate_priority(job));

    job->inc_deficit_counter(new_job_deficit_);

    job_queue_.push(job);

    ++num_jobs_;

    schedule_job();
}

void Scheduler::schedule_job() {
#ifdef PRINT_SCHEDULE_TIME
    auto start_schedule_time = std::chrono::steady_clock::now();
    static unsigned num_scheduled_stages = 0;
    static double total_schedule_time = 0;
    constexpr unsigned schedule_time_print_interval = 100000;
    static unsigned schedule_time_next_print = schedule_time_print_interval;
#endif

    if (cuda_streams_.empty()) {
        return;
    }

#ifdef PRINT_SCHEDULE_TIME
    bool has_scheduled = false;
#endif

#ifdef PRINT_QUEUE_SIZE
    static unsigned total_queue_size = 0;
    static unsigned queue_size_i = 0;
    constexpr unsigned queue_size_print_interval = 100000;

    if (!job_queue_.empty()) {
        total_queue_size += job_queue_.size();
        ++queue_size_i;
    }

    if (queue_size_i >= queue_size_print_interval) {
        printf("queue size: %lf\n", (double)total_queue_size / queue_size_i);
        queue_size_i = 0;
        total_queue_size = 0;
    }
#endif

    while (!job_queue_.empty()) {
        job::Job* job = job_queue_.top();
        job_queue_.pop();

#ifdef PRINT_SCHEDULE_TIME
        ++num_scheduled_stages;
        has_scheduled = true;
#endif
        if (!job->has_started()) {
            job->set_started();
        }

        job->set_running(cuda_streams_.back());
        cuda_streams_.pop_back();

        bool job_is_mem = job->is_mem();

        bool fits;
        if (job_is_mem) {
            fits = true;
        } else {
            fits = job_fits(job);
        }

        if (!fits) {
            break;
        }

        if (job->is_pre_notify()) {
            server_->notify_job_starts(job);
        }

#ifdef PRINT_NUM_RUNNING_KERNELS
        ++num_running_kernels_;
        printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif

        job::Context::set_current_job(job);
        job->run_next();

        // Note: after run_next, the is_mem flag may change, but we want to use the old one
        if (job_is_mem) {
            cudaLaunchHostFunc(job->get_cuda_stream(), mem_notification_callback, job);
        }

        if (job->has_next()) {
            server_->set_job_stage_resource(job, job->get_cur_stage() + 1, normalize_resources(job));
            job->set_priority(calculate_priority(job));
        }

        if (!fits || cuda_streams_.empty()) {
            break;
        }
    }

#ifdef PRINT_SCHEDULE_TIME
    if (has_scheduled) {
        auto end_schedule_time = std::chrono::steady_clock::now();
        total_schedule_time += std::chrono::duration<double, std::micro>(end_schedule_time - start_schedule_time).count();
        if (num_scheduled_stages >= schedule_time_next_print) {
            printf("Schedule time, # stages, time / stage: %lf %u %lf\n", total_schedule_time, num_scheduled_stages, total_schedule_time / num_scheduled_stages);
            schedule_time_next_print += schedule_time_print_interval;
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
        total_sm_avail_.minus(job, 1);

        // TODO: handle overusing resources

        job->add_predicted_smid(smid);
    }
}

void Scheduler::update_deficit_counters(job::Job* job_scheduled) {
    job_scheduled->inc_deficit_counter(-1);
    new_job_deficit_ -= 1. / num_jobs_;

    // To avoid overflow
    if (new_job_deficit_ < 10. - DBL_MAX) {
        for (const auto& job : job_id_to_job_map_) {
            job->inc_deficit_counter(-new_job_deficit_);
        }
        new_job_deficit_ = 0;
    }
}

float Scheduler::normalize_resources(job::Job* job) {
    return ((float)(job->get_num_registers_per_thread() * job->get_num_threads_per_block()) / total_nregs_ + (float)job->get_num_threads_per_block() / total_nthrs_ + (float)job->get_smem_size_per_block() / total_smem_) * job->get_num_blocks() + (float)job->get_num_blocks() / total_nblocks_;
}

double Scheduler::calculate_priority(job::Job* job) const {
    return calculate_packing(job) - eta_ * server_->get_job_remaining_rl(job, job->get_cur_stage() + 1);
}

double Scheduler::calculate_packing(job::Job* job) const {
    return total_sm_avail_.dot(job) / max_resources_dot_prod_;
}

void mem_notification_callback(void* job) {
    ipc::ShmChannel* channel = job::Context::get_mem2sched_channel();
    channel->write(job);
}

}
}

