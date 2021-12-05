#include <llis/server/scheduler_full.h>

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

SchedulerFull::SchedulerFull(float unfairness_threshold, float eta) :
        server_(nullptr),
        gpu2sched_channel_(GPU2SCHED_CHAN_SIZE),
#ifdef LLIS_MEASURE_BLOCK_TIME
        gpu2sched_block_time_channel_(GPU2SCHED_CHAN_SIZE_TIME),
#endif
        mem2sched_channel_(10240),
        cuda_streams_(500),
        unfairness_threshold_(unfairness_threshold),
        eta_(eta),
        job_less_(unfairness_threshold_) { // TODO: size of the channel must be larger than number of total blocks * 2
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

void SchedulerFull::set_server(Server* server) {
    server_ = server;
    profiler_ = server_->get_profiler();
}

void SchedulerFull::try_handle_block_start_finish() {
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

void SchedulerFull::handle_block_start_finish() {
    job::InstrumentInfo info = gpu2sched_channel_.read<job::InstrumentInfo>();

    if (info.is_start) {
        handle_block_start(info);
    } else {
        handle_block_finish(info);
    }
}

#ifdef LLIS_MEASURE_BLOCK_TIME
void SchedulerFull::handle_block_start_end_time() {
    job::BlockStartEndTime start_end_time = gpu2sched_block_time_channel_.read<job::BlockStartEndTime>();

    uint32_t start = (uint32_t)start_end_time.data[0] << 8 | start_end_time.data[1] >> 8;
    uint32_t end = (uint32_t)(start_end_time.data[1] & 0xFF) << 16 | start_end_time.data[2];
    profiler_->record_block_exec_time(start, end);
}
#endif

void SchedulerFull::handle_block_start(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id].get();

    if (!job->has_predicted_smid(info.smid)) {
        gpu_resources_.acquire(info.smid, job, 1);
    } else {
        job->dec_predicted_smid(info.smid);
    }

    if (!job->mark_block_start()) {
        const unsigned* predicted_smid_nums = job->get_predicted_smid_nums();
        unsigned total_wrong_prediction = 0;
        unsigned total_wrong_prediction_sm = 0;
        for (unsigned smid = 0; smid < gpu_resources_.get_num_sms(); ++smid) {
            unsigned num = predicted_smid_nums[smid];

            if (num > 0) {
                total_wrong_prediction += num;
                ++total_wrong_prediction_sm;
                gpu_resources_.release(info.smid, job, num);
            }
        }
        profiler_->recrod_kernel_block_mis_alloc(job->get_cur_num_blocks(), total_wrong_prediction, total_wrong_prediction_sm);

        if (job->is_unfit()) {
            if (num_outstanding_kernels_ > 0) {
                --num_outstanding_kernels_;
            }
            job->unset_unfit();
        }

        schedule_job();
    }
}

void SchedulerFull::handle_block_finish(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id].get();

    job->mark_block_finish();
    if (!job->is_running()) {
        if (job->has_next()) {
            if (job->is_mem()) {
                mem_job_queue_.push(job);
            } else {
                job_queue_.push_back(job);
            }
        } else {
            server_->notify_job_ends(job);
            --num_jobs_;
        }

        auto end_time = std::chrono::steady_clock::now();
        auto start_time = job->get_stage_start_time();
#ifdef LLIS_ENABLE_PROFILER
        profiler_->record_kernel_info(start_time, end_time, job->get_cur_num_blocks(), job->get_cur_num_threads_per_block(), job->get_cur_smem_size_per_block(), job->get_cur_num_registers_per_thread(), job->get_priority(), job->get_registered_job_id());
#endif
        double length = std::chrono::duration<double, std::micro>(end_time - start_time).count();
        server_->update_job_stage_length(job, job->get_cur_stage(), length);

#ifdef PRINT_NUM_RUNNING_KERNELS
        --num_running_kernels_;
        printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif

        cuda_streams_.push_back(job->get_cuda_stream());
        finished_block_notifiers_.push_back(job->get_finished_block_notifier());
    }

    gpu_resources_.release(info.smid, job, 1);

    if (!job->is_running() && !job->has_next()) {
        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_id_to_job_map_[info.job_id]));
    }

    schedule_job();
}

void SchedulerFull::handle_mem_finish() {
    job::Job* job;
    mem2sched_channel_.read(&job);

    if (job->has_next()) {
        if (job->is_mem()) {
            mem_job_queue_.push(job);
        } else {
            job_queue_.push_back(job);
        }
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

    has_mem_job_running_ = false;

    schedule_job();
}

void SchedulerFull::handle_new_job(std::unique_ptr<job::Job> job_) {
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

    server_->set_job_stage_resource(job, job->get_cur_stage() + 1, gpu_resources_.normalize_resources(job));

    job->inc_deficit_counter(new_job_deficit_);

    if (job->is_mem()) {
        mem_job_queue_.push(job);
    } else {
        job_queue_.push_back(job);
    }

    ++num_jobs_;

    schedule_job();
}

void SchedulerFull::schedule_job() {
    schedule_mem_job();
    schedule_comp_job();
}

void SchedulerFull::schedule_comp_job() {
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

#ifdef PRINT_SORT_TIME
    constexpr unsigned sort_time_next_print = 100000;
    static unsigned sort_time_i = 0;
    auto start_sort_time = std::chrono::steady_clock::now();
#endif
    for (job::Job* job : job_queue_) {
        job->set_priority(calculate_priority(job));
    }
    std::sort(job_queue_.begin(), job_queue_.end(), job_less_);
#ifdef PRINT_SORT_TIME
    if (sort_time_i >= sort_time_next_print) {
        auto end_sort_time = std::chrono::steady_clock::now();
        double time_taken_to_sort = std::chrono::duration<double, std::micro>(end_sort_time - start_sort_time).count();
        printf("Sort time: %lf\n", time_taken_to_sort);
        sort_time_i = 0;
    }
    ++sort_time_i;
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

    while (!job_queue_.empty()) {
        job::Job* job = job_queue_.back();

        if (!gpu_resources_.job_fits(job)) {
            if (num_outstanding_kernels_ >= max_num_outstanding_kernels_) {
                break;
            } else {
                ++num_outstanding_kernels_;
                job->set_unfit();
            }
        }

        job_queue_.pop_back();

#ifdef PRINT_SCHEDULE_TIME
        ++num_scheduled_stages;
        has_scheduled = true;
#endif
        if (!job->has_started()) {
            job->set_started();
        }

        job->set_running(cuda_streams_.back());
        cuda_streams_.pop_back();
        job->set_finished_block_notifier(finished_block_notifiers_.back());
        finished_block_notifiers_.pop_back();

        gpu_resources_.choose_sms(job);

        if (job->is_pre_notify()) {
            server_->notify_job_starts(job);
        }

#ifdef PRINT_NUM_RUNNING_KERNELS
        ++num_running_kernels_;
        printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif

        job::Context::set_current_job(job);

        auto start_run_next_time = std::chrono::steady_clock::now();
        job->run_next();
        auto end_run_next_time = std::chrono::steady_clock::now();
        profiler_->record_run_next_time(start_run_next_time, end_run_next_time, job->get_cur_num_blocks());

        if (job->has_next()) {
            server_->set_job_stage_resource(job, job->get_cur_stage() + 1, gpu_resources_.normalize_resources(job));
        }

        if (cuda_streams_.empty()) {
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

void SchedulerFull::schedule_mem_job() {
    if (has_mem_job_running_ || cuda_streams_.empty() || mem_job_queue_.empty()) {
        return;
    }

    job::Job* job = mem_job_queue_.front();
    mem_job_queue_.pop();

    if (!job->has_started()) {
        job->set_started();
    }

    job->set_running(cuda_streams_.back());
    cuda_streams_.pop_back();

    if (job->is_pre_notify()) {
        server_->notify_job_starts(job);
    }

    job::Context::set_current_job(job);
    job->run_next();

    cudaLaunchHostFunc(job->get_cuda_stream(), mem_notification_callback, job);

    if (job->has_next()) {
        server_->set_job_stage_resource(job, job->get_cur_stage() + 1, gpu_resources_.normalize_resources(job));
    }

    has_mem_job_running_ = true;
}

void SchedulerFull::update_deficit_counters(job::Job* job_scheduled) {
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

double SchedulerFull::calculate_priority(job::Job* job) const {
    return calculate_packing(job) - eta_ * server_->get_job_remaining_rl(job, job->get_cur_stage() + 1);
}

double SchedulerFull::calculate_packing(job::Job* job) const {
    return gpu_resources_.dot_normalized(job);
}

void mem_notification_callback(void* job) {
    ipc::ShmChannelCpuWriter* channel = job::Context::get_mem2sched_channel();
    channel->write(job);
}

}
}
