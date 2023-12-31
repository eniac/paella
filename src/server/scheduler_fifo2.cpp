#include <llis/server/scheduler_fifo2.h>

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/server/server.h>
#include <llis/job/job.h>
#include <llis/job/context.h>
#include <llis/job/instrument_info.h>

#include <spdlog/spdlog.h>

#include <sys/socket.h>
#include <sys/un.h>

#include <memory>
#include <string>
#include <algorithm>
#include <chrono>
#include <cfloat>

namespace llis {
namespace server {

SchedulerFifo2::SchedulerFifo2(unsigned num_streams, unsigned sched_sleep) :
        gpu2sched_channel_(GPU2SCHED_CHAN_SIZE),
#ifdef LLIS_MEASURE_BLOCK_TIME
        gpu2sched_block_time_channel_(GPU2SCHED_CHAN_SIZE_TIME),
#endif
        mem2sched_channel_(10240),
        cuda_streams_(num_streams) {
    SPDLOG_INFO("Setting up LLIS FIFO2 scheduler...");
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

void SchedulerFifo2::try_handle_block_start_finish() {
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

void SchedulerFifo2::handle_block_start_finish() {
    job::InstrumentInfo info = gpu2sched_channel_.read<job::InstrumentInfo>();

    if (info.is_start) {
        handle_block_start(info);
    } else {
        handle_block_finish(info);
    }
}

#ifdef LLIS_MEASURE_BLOCK_TIME
void SchedulerFifo2::handle_block_start_end_time() {
    job::BlockStartEndTime start_end_time = gpu2sched_block_time_channel_.read<job::BlockStartEndTime>();

    uint32_t start = (uint32_t)start_end_time.data[0] << 8 | start_end_time.data[1] >> 8;
    uint32_t end = (uint32_t)(start_end_time.data[1] & 0xFF) << 16 | start_end_time.data[2];
    profiler_->record_block_exec_time(start, end);
}
#endif

void SchedulerFifo2::handle_block_start(const job::InstrumentInfo& info) {
}

void SchedulerFifo2::handle_block_finish(const job::InstrumentInfo& info) {
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
        if (job->has_next()) {
            job_queue_.push(job);
        } else {
#ifdef LLIS_ENABLE_PROFILER
            profiler_->record_job_event(job->get_id(), Profiler::JobEvent::JOB_FINISHED);
#endif
            server_->notify_job_ends(job);
            --num_jobs_;
        }

        auto end_time = std::chrono::steady_clock::now();
        auto start_time = job->get_stage_start_time();
#ifdef LLIS_ENABLE_PROFILER
        profiler_->record_kernel_info(start_time, end_time, job->get_cur_num_blocks(), job->get_cur_num_threads_per_block(), job->get_cur_smem_size_per_block(), job->get_cur_num_registers_per_thread(), job->get_priority(), job->get_registered_job_id());
#endif
        double length = std::chrono::duration<double, std::micro>(end_time - start_time).count();

#ifdef PRINT_NUM_RUNNING_KERNELS
        --num_running_kernels_;
        printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif

        cuda_streams_.push_back(job->get_cuda_stream());
        finished_block_notifiers_.push_back(job->get_finished_block_notifier());
    }

    if (!job->is_running() && !job->has_next()) {
        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_id_to_job_map_[info.job_id]));
    }

    schedule_job();
}

void SchedulerFifo2::handle_mem_finish() {
    job::Job* job;
    mem2sched_channel_.read(&job);

    if (job->has_next()) {
        job_queue_.push(job);
    } else {
#ifdef LLIS_ENABLE_PROFILER
        profiler_->record_job_event(job->get_id(), Profiler::JobEvent::JOB_FINISHED);
#endif
        server_->notify_job_ends(job);
        --num_jobs_;
    }

    job->unset_running();

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

void SchedulerFifo2::handle_new_job(std::unique_ptr<job::Job> job_) {
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

    job_queue_.push(job);

    ++num_jobs_;

    schedule_job();
}

void SchedulerFifo2::schedule_job() {
    if (cuda_streams_.empty() || job_queue_.empty()) {
        return;
    }

    while (!job_queue_.empty()) {
        job::Job* job = job_queue_.top();
        job_queue_.pop();

#ifdef LLIS_ENABLE_PROFILER
        profiler_->record_job_event(job->get_id(), Profiler::JobEvent::KERNEL_SCHED_START);
#endif

        if (!job->has_started()) {
            job->set_started();
        }

        job->set_running(cuda_streams_.back());
        cuda_streams_.pop_back();

        bool is_mem = job->is_mem();

        if (!is_mem) {
            job->set_finished_block_notifier(finished_block_notifiers_.back());
            finished_block_notifiers_.pop_back();
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
        profiler_->record_run_next_time(start_run_next_time, end_run_next_time, is_mem ? 0 : job->get_cur_num_blocks());
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
    }
}

void SchedulerFifo2::mem_notification_callback(void* job) {
    ipc::ShmChannelCpuWriter* channel = job::Context::get_mem2sched_channel();
    channel->write(job);
}

LLIS_SCHEDULER_REGISTER("fifo2", [](const po::variables_map& args) -> std::unique_ptr<Scheduler> {
        unsigned num_streams = args["num_streams"].as<unsigned>();
        unsigned sched_sleep = args["sched_sleep"].as<unsigned>();
        return std::make_unique<SchedulerFifo2>(num_streams, sched_sleep);
    });

}
}
