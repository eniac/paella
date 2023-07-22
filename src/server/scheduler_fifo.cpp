#include <llis/server/scheduler_fifo.h>

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/server/server.h>
#include <llis/job/job.h>
#include <llis/job/context.h>
#include <llis/job/instrument_info.h>
#include <llis/utils/error.h>

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

void mem_notification_callback(void* job);

SchedulerFifo::SchedulerFifo(unsigned num_streams, unsigned sched_sleep) :
        gpu2sched_channel_(GPU2SCHED_CHAN_SIZE),
#ifdef LLIS_MEASURE_BLOCK_TIME
        gpu2sched_block_time_channel_(GPU2SCHED_CHAN_SIZE_TIME),
#endif
        mem2sched_channel_(10240),
        cuda_streams_(num_streams) { // TODO: size of the channel must be larger than number of total blocks * 2
    SPDLOG_INFO("Setting up LLIS FIFO scheduler...");
    job::Context::set_gpu2sched_channel(&gpu2sched_channel_);
#ifdef LLIS_MEASURE_BLOCK_TIME
    job::Context::set_gpu2sched_block_time_channel(&gpu2sched_block_time_channel_);
#endif
    job::Context::set_mem2sched_channel(&mem2sched_channel_);

    for (auto& stream : cuda_streams_) {
        utils::error_throw_cuda(cudaStreamCreate(&stream));
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

void SchedulerFifo::try_handle_block_start_finish() {
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

void SchedulerFifo::handle_block_start_finish() {
    job::InstrumentInfo info = gpu2sched_channel_.read<job::InstrumentInfo>();

    if (info.is_start) {
        handle_block_start(info);
    } else {
        handle_block_finish(info);
    }
}

void SchedulerFifo::handle_block_start(const job::InstrumentInfo& info) {
}

void SchedulerFifo::handle_block_finish(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id].get();

#ifdef LLIS_FINISHED_BLOCK_NOTIFICATION_AGG
    remaining_num_blocks_[info.job_id] -= info.num;
    pre_notify_blocks_[job->get_id()] -= info.num;
#ifdef PRINT_NUM_RUNNING_BLOCKS
    num_running_blocks_ -= info.num;
    printf("num_running_blocks_: %u, deducted %u\n", num_running_blocks_, info.num);
#endif
#else
    --remaining_num_blocks_[info.job_id];
    --pre_notify_blocks_[job->get_id()];
#ifdef PRINT_NUM_RUNNING_BLOCKS
    --num_running_blocks_;
    printf("num_running_blocks_: %u, deducted 1\n", num_running_blocks_);
#endif
#endif

    if (pre_notify_blocks_[job->get_id()] <= 0 && !pre_notify_sent_[job->get_id()]) {
        server_->notify_job_starts(job);
        pre_notify_sent_[job->get_id()] = true;
    }
    if (remaining_num_blocks_[info.job_id] == 0) {
        server_->notify_job_ends(job);

        cuda_streams_.push_back(job->get_cuda_stream());
        finished_block_notifiers_.push_back(job->get_finished_block_notifier());

        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_id_to_job_map_[info.job_id]));

        --num_jobs_;

#ifdef PRINT_NUM_RUNNING_JOBS
        --num_running_jobs_;
        printf("num_running_jobs_: %u\n", num_running_jobs_);
#endif
    }

    schedule_job();
}

void SchedulerFifo::handle_mem_finish() {
    job::Job* job;
    mem2sched_channel_.read(&job);

    --remaining_num_blocks_[job->get_id()];
    --pre_notify_blocks_[job->get_id()];

    if (pre_notify_blocks_[job->get_id()] <= 0 && !pre_notify_sent_[job->get_id()]) {
        server_->notify_job_starts(job);
        pre_notify_sent_[job->get_id()] = true;
    }
    if (remaining_num_blocks_[job->get_id()] == 0) {
        server_->notify_job_ends(job);

        cuda_streams_.push_back(job->get_cuda_stream());
        finished_block_notifiers_.push_back(job->get_finished_block_notifier());

        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_id_to_job_map_[job->get_id()]));

        --num_jobs_;

#ifdef PRINT_NUM_RUNNING_JOBS
        --num_running_jobs_;
        printf("num_running_jobs_: %u\n", num_running_jobs_);
#endif
    }

    schedule_job();
}

#ifdef LLIS_MEASURE_BLOCK_TIME
void SchedulerFifo::handle_block_start_end_time() {
    job::BlockStartEndTime start_end_time = gpu2sched_block_time_channel_.read<job::BlockStartEndTime>();

    uint32_t start = (uint32_t)start_end_time.data[0] << 8 | start_end_time.data[1] >> 8;
    uint32_t end = (uint32_t)(start_end_time.data[1] & 0xFF) << 16 | start_end_time.data[2];
    profiler_->record_block_exec_time(start, end);
}
#endif

void SchedulerFifo::handle_new_job(std::unique_ptr<job::Job> job_) {
    job::Job* job = job_.get();
    if (!job->has_next()) {
        unused_job_id_.push_back(job->get_id());
        server_->release_job_instance(std::move(job_));
        return;
    }

    if (unused_job_id_.empty()) {
        job->set_id(job_id_to_job_map_.size());
        job_id_to_job_map_.push_back(std::move(job_));
        remaining_num_blocks_.push_back(0);
        pre_notify_blocks_.push_back(0);
        pre_notify_sent_.push_back(false);
    } else {
        job->set_id(unused_job_id_.back());
        unused_job_id_.pop_back();
        job_id_to_job_map_[job->get_id()] = std::move(job_);
        remaining_num_blocks_[job->get_id()] = 0;
        pre_notify_blocks_[job->get_id()] = 0;
        pre_notify_sent_[job->get_id()] = false;
    }

    job_queue_.push(job);

    ++num_jobs_;

    schedule_job();
}

void SchedulerFifo::schedule_job() {
    if (cuda_streams_.empty() || job_queue_.empty()) {
        return;
    }

    while (!job_queue_.empty()) {
        job::Job* job = job_queue_.front();
        job_queue_.pop();

        job->set_started();

        job->set_running(cuda_streams_.back());
        cuda_streams_.pop_back();
        job->set_finished_block_notifier(finished_block_notifiers_.back());
        finished_block_notifiers_.pop_back();

#ifdef PRINT_NUM_RUNNING_JOBS
        ++num_running_jobs_;
        printf("num_running_jobs_: %u\n", num_running_jobs_);
#endif

        job::Context::set_current_job(job);

        bool saw_pre_notify = false;

        while (job->has_next()) {
            bool is_mem = job->is_mem();
            if (is_mem) {
                ++remaining_num_blocks_[job->get_id()];
                if (!saw_pre_notify) {
                    ++pre_notify_blocks_[job->get_id()];
                    if (job->is_pre_notify()) {
                        saw_pre_notify = true;
                    }
                }
            } else {
                remaining_num_blocks_[job->get_id()] += job->get_num_blocks();
                if (!saw_pre_notify) {
                    pre_notify_blocks_[job->get_id()] += job->get_num_blocks();
                    if (job->is_pre_notify()) {
                        saw_pre_notify = true;
                    }
                }
#ifdef PRINT_NUM_RUNNING_BLOCKS
                num_running_blocks_ += job->get_num_blocks();
                printf("num_running_blocks_: %u\n", num_running_blocks_);
#endif
            }
            job->run_next();
            if (is_mem) {
                cudaLaunchHostFunc(job->get_cuda_stream(), mem_notification_callback, job);
            }
        }

        if (cuda_streams_.empty()) {
            break;
        }
    }
}


void SchedulerFifo::mem_notification_callback(void* job) {
    ipc::ShmChannelCpuWriter* channel = job::Context::get_mem2sched_channel();
    channel->write(job);
}

LLIS_SCHEDULER_REGISTER("fifo", [](const po::variables_map& args) -> std::unique_ptr<Scheduler> {
        unsigned num_streams = args["num_streams"].as<unsigned>();
        unsigned sched_sleep = args["sched_sleep"].as<unsigned>();
        return std::make_unique<SchedulerFifo>(num_streams, sched_sleep);
    });

}
}
