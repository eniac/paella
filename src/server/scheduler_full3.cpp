#include <llis/server/scheduler_full3.h>

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

SchedulerFull3::SchedulerFull3(float unfairness_threshold, float eta) :
        server_(nullptr),
        gpu2sched_channel_(GPU2SCHED_CHAN_SIZE),
#ifdef LLIS_MEASURE_BLOCK_TIME
        gpu2sched_block_time_channel_(GPU2SCHED_CHAN_SIZE_TIME),
#endif
        mem2sched_channel_(409600),
        cuda_streams_(32),
        eta_(eta),
        job_queue_(unfairness_threshold) { // TODO: size of the channel must be larger than number of total blocks * 2
    LLIS_INFO("Setting up LLIS scheduler...");
    job::Context::set_gpu2sched_channel(&gpu2sched_channel_);
#ifdef LLIS_MEASURE_BLOCK_TIME
    job::Context::set_gpu2sched_block_time_channel(&gpu2sched_block_time_channel_);
#endif
    job::Context::set_mem2sched_channel(&mem2sched_channel_);

    for (auto& stream : cuda_streams_) {
        cudaStreamCreate(&stream);
    }

    finished_block_notifiers_raw_ = job::FinishedBlockNotifier::create_array(cuda_streams_.size(), &gpu2sched_channel_
#ifdef LLIS_MEASURE_BLOCK_TIME
        , &gpu2sched_block_time_channel_
#endif
    );
    for (unsigned i = 0; i < cuda_streams_.size(); ++i) {
        finished_block_notifiers_.push_back(finished_block_notifiers_raw_ + i);
    }
}

void SchedulerFull3::set_server(Server* server) {
    server_ = server;
    profiler_ = server_->get_profiler();
}

void SchedulerFull3::try_handle_block_start_finish() {
    if (gpu2sched_channel_.can_read<job::InstrumentInfo>()) {
        handle_block_start_finish();
    }

#ifdef LLIS_MEASURE_BLOCK_TIME
    if (gpu2sched_block_time_channel_.can_read<job::BlockStartEndTime>()) {
        handle_block_start_end_time();
    }
#endif

    if (mem2sched_channel_.can_read()) {
        handle_mem_finish();
    }
}

void SchedulerFull3::handle_block_start_finish() {
    job::InstrumentInfo info = gpu2sched_channel_.read<job::InstrumentInfo>();

    if (info.is_start) {
        handle_block_start(info);
    } else {
        handle_block_finish(info);
    }
}

#ifdef LLIS_MEASURE_BLOCK_TIME
void SchedulerFull3::handle_block_start_end_time() {
    job::BlockStartEndTime start_end_time = gpu2sched_block_time_channel_.read<job::BlockStartEndTime>();

#ifdef LLIS_ENABLE_PROFILER
    uint32_t start = (uint32_t)start_end_time.data[0] << 8 | start_end_time.data[1] >> 8;
    uint32_t end = (uint32_t)(start_end_time.data[1] & 0xFF) << 16 | start_end_time.data[2];
    profiler_->record_block_exec_time(start, end);
#endif
}
#endif

void SchedulerFull3::handle_block_start(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id].get();

#ifdef LLIS_FINISHED_BLOCK_NOTIFICATION_AGG
    if (!job->mark_block_start(info.num)) {
#else
    if (!job->mark_block_start()) {
#endif
#ifdef PRINT_NUM_NOTIF
        ++num_start_notif_received;
        printf("num_start_notif_received: %u\n", num_start_notif_received);
#endif
        if (job->is_unfit()) {
            job->unset_unfit();
            --num_outstanding_kernels_;
#ifdef PRINT_NUM_OUTSTANDING_KERNELS
            printf("num_outstanding_kernels_: %u\n", num_outstanding_kernels_);
#endif
            schedule_job();
        }
    }
}

void SchedulerFull3::handle_block_finish(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id].get();

#ifdef LLIS_FINISHED_BLOCK_NOTIFICATION_AGG
    job->mark_block_finish(info.num);
#else
    job->mark_block_finish();
#endif
    if (!job->is_running()) {
#ifdef LLIS_ENABLE_PROFILER
        profiler_->record_job_event(job->get_id(), Profiler::JobEvent::KERNEL_FINISHED);
#endif
        auto end_time = std::chrono::steady_clock::now();
        auto start_time = job->get_stage_start_time();
#ifdef LLIS_ENABLE_PROFILER
        profiler_->record_kernel_info(start_time, end_time, job->get_cur_num_blocks(), job->get_cur_num_threads_per_block(), job->get_cur_smem_size_per_block(), job->get_cur_num_registers_per_thread(), job->get_priority(), job->get_registered_job_id());
#endif
        double length = std::chrono::duration<double, std::micro>(end_time - start_time).count();
        server_->update_job_stage_length(job, job->get_cur_stage(), length);

        if (job->has_next()) {
            if (!server_->has_job_stage_resource(job, job->get_cur_stage() + 1)) {
                server_->set_job_stage_resource(job, job->get_cur_stage() + 1, job->is_mem() ? 0.1 : gpu_resources_.normalize_resources(job) * job->get_num_blocks());
            }
            job->set_priority(calculate_priority(job));

            job_queue_.push(job);
        } else {
#ifdef LLIS_ENABLE_PROFILER
            profiler_->record_job_event(job->get_id(), Profiler::JobEvent::JOB_FINISHED);
#endif
            server_->notify_job_ends(job);
            --num_jobs_;
        }

#ifdef PRINT_NUM_RUNNING_KERNELS
        --num_running_kernels_;
        printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif
#ifdef PRINT_NUM_NOTIF
        ++num_end_notif_received;
        printf("num_end_notif_received: %u\n", num_end_notif_received);
#endif

        // This check is a workaround of a bug in CUDA atomic
        if (job->is_unfit()) {
            job->unset_unfit();
            --num_outstanding_kernels_;
#ifdef PRINT_NUM_OUTSTANDING_KERNELS
            printf("num_outstanding_kernels_: %u\n", num_outstanding_kernels_);
#endif
        }

        cuda_streams_.push_back(job->get_cuda_stream());
        finished_block_notifiers_.push_back(job->get_finished_block_notifier());
    }

#ifdef LLIS_FINISHED_BLOCK_NOTIFICATION_AGG
    gpu_resources_.release(job, info.num);
#else
    gpu_resources_.release(job, 1);
#endif

    if (!job->is_running() && !job->has_next()) {
        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_id_to_job_map_[info.job_id]));
    }

    schedule_job();
}

void SchedulerFull3::handle_mem_finish() {
    job::Job* job;
    mem2sched_channel_.read(&job);

    job->unset_running();

    auto end_time = std::chrono::steady_clock::now();
    auto start_time = job->get_stage_start_time();
    double length = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    server_->update_job_stage_length(job, job->get_cur_stage(), length);

    if (job->has_next()) {
        if (!server_->has_job_stage_resource(job, job->get_cur_stage() + 1)) {
            server_->set_job_stage_resource(job, job->get_cur_stage() + 1, job->is_mem() ? 0.1 : gpu_resources_.normalize_resources(job) * job->get_num_blocks());
        }
        job->set_priority(calculate_priority(job));

        job_queue_.push(job);
    } else {
#ifdef LLIS_ENABLE_PROFILER
        profiler_->record_job_event(job->get_id(), Profiler::JobEvent::JOB_FINISHED);
#endif
        server_->notify_job_ends(job);
        --num_jobs_;
    }

#ifdef PRINT_NUM_RUNNING_KERNELS
    --num_running_mems_;
    printf("num_running_mems_: %u\n", num_running_mems_);
#endif

    cuda_streams_.push_back(job->get_cuda_stream());

    if (!job->has_next()) {
        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_id_to_job_map_[job->get_id()]));
    }

    schedule_job();
}

void SchedulerFull3::handle_new_job(std::unique_ptr<job::Job> job_) {
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
#ifdef LLIS_ENABLE_PROFILER
    profiler_->record_job_event(job->get_id(), Profiler::JobEvent::JOB_SUBMITTED);
#endif

    if (!server_->has_job_stage_resource(job, job->get_cur_stage() + 1)) {
        server_->set_job_stage_resource(job, job->get_cur_stage() + 1, job->is_mem() ? 0.1 : gpu_resources_.normalize_resources(job) * job->get_num_blocks());
    }
    job->set_stage_lengths_resources(server_->get_job_remaining_rl(job, 0), server_->get_job_stage_lengths(job), server_->get_job_stage_resources(job));
    job->set_priority(calculate_priority(job));

    job_queue_.push(job);

    ++num_jobs_;

    schedule_job();
}

void SchedulerFull3::schedule_job() {
    if (cuda_streams_.empty() || job_queue_.empty()) {
        return;
    }

#ifdef PRINT_SCHEDULE_TIME
    auto start_schedule_time = std::chrono::steady_clock::now();
    static unsigned num_scheduled_stages = 0;
    static double total_schedule_time = 0;
    constexpr unsigned schedule_time_print_interval = 100000;
    static unsigned schedule_time_next_print = schedule_time_print_interval;
#endif

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

    do {
        job::Job* job = job_queue_.top();

#ifdef LLIS_ENABLE_PROFILER
        profiler_->record_job_event(job->get_id(), Profiler::JobEvent::KERNEL_SCHED_START);
#endif

        bool is_mem = job->is_mem();

        if (!is_mem && !gpu_resources_.job_fits(job)) {
            if (num_outstanding_kernels_ >= max_num_outstanding_kernels_) {
#ifdef LLIS_ENABLE_PROFILER
                profiler_->record_job_event(job->get_id(), Profiler::JobEvent::KERNEL_SCHED_ABORT);
#endif
                break;
            } else {
                ++num_outstanding_kernels_;
#ifdef PRINT_NUM_OUTSTANDING_KERNELS
                printf("num_outstanding_kernels_: %u\n", num_outstanding_kernels_);
#endif
                job->set_unfit();
            }
        }

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

        if (!is_mem) {
            job->set_finished_block_notifier(finished_block_notifiers_.back());
            finished_block_notifiers_.pop_back();

            gpu_resources_.acquire(job, job->get_cur_num_blocks());
        }

        if (job->is_pre_notify()) {
            server_->notify_job_starts(job);
        }

#ifdef PRINT_NUM_RUNNING_KERNELS
        if (is_mem) {
            ++num_running_mems_;
            printf("num_running_mems_: %u\n", num_running_mems_);
        } else {
            ++num_running_kernels_;
            printf("num_running_kernels_: %u\n", num_running_kernels_);
            ++num_scheduled_kernels_;
            printf("num_scheduled_kernels_: %u\n", num_scheduled_kernels_);
        }
#endif

        job::Context::set_current_job(job);

#ifdef LLIS_ENABLE_PROFILER
        auto start_run_next_time = std::chrono::steady_clock::now();
        profiler_->record_job_event(job->get_id(), Profiler::JobEvent::KERNEL_SUBMIT_START);
#endif
        job->run_next();
#ifdef LLIS_ENABLE_PROFILER
        auto end_run_next_time = std::chrono::steady_clock::now();
        profiler_->record_run_next_time(start_run_next_time, end_run_next_time, job->get_cur_num_blocks());
#endif

        if (is_mem) {
            cudaLaunchHostFunc(job->get_cuda_stream(), mem_notification_callback, job);
        }

#ifdef LLIS_ENABLE_PROFILER
        profiler_->record_job_event(job->get_id(), Profiler::JobEvent::KERNEL_SUBMIT_END);
#endif

        if (cuda_streams_.empty()) {
            break;
        }
    } while (!job_queue_.empty());

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

double SchedulerFull3::calculate_priority(job::Job* job) const {
    double cur_rl_ = job->get_cur_rl();
    job->dec_cur_rl();
    return -cur_rl_;
    //return -server_->get_job_remaining_length(job, job->get_cur_stage() + 1);
}

double SchedulerFull3::calculate_packing(job::Job* job) const {
    return gpu_resources_.dot_normalized(job);
}

void SchedulerFull3::mem_notification_callback(void* job) {
    ipc::ShmChannelCpuWriter* channel = job::Context::get_mem2sched_channel();
    channel->write(job);
}

}
}
