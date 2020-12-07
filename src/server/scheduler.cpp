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

namespace llis {
namespace server {

Scheduler::Scheduler() : server_(nullptr), gpu2sched_channel_(10240), cuda_streams_(100) { // TODO: size of the channel must be larger than number of total blocks * 2
    job::Context::set_gpu2sched_channel(&gpu2sched_channel_);

    for (auto& stream : cuda_streams_) {
        cudaStreamCreate(&stream);
    }

    // TODO: query the numbers
    // Some of them can be zero because some smid may not reflect an actual SM
    sm_avails_.resize(56);
    for (auto& sm_avail : sm_avails_) {
        sm_avail.nregs = 65536;
        sm_avail.nthrs = 2048;
        sm_avail.smem = 65536;
        sm_avail.nblocks = 32;
    }
}

void Scheduler::set_server(Server* server) {
    server_ = server;
}

void Scheduler::try_handle_block_start_finish() {
    if (gpu2sched_channel_.can_read<job::InstrumentInfo>()) {
        handle_block_start_finish();
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

    // TODO: handle allocation granularity
    sm_avails_[info.smid].nregs -= job->get_num_registers_per_thread() * job->get_num_threads_per_block(); // TODO: use an actual number
    sm_avails_[info.smid].nthrs -= job->get_num_threads_per_block();
    sm_avails_[info.smid].smem -= job->get_smem_size_per_block();
}

void Scheduler::handle_block_finish(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id];

    job->mark_block_finish();
    if (!job->is_running()) {
        if (!job->has_next()) {
            server_->notify_job_ends(job);
        }
        cuda_streams_.push_back(job->get_cuda_stream());
    }

    // TODO: handle resouce release

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

    jobs_.push_back(std::move(job));

    schedule_job();
}

void Scheduler::schedule_job() {
    while (!jobs_.empty() && !jobs_.front()->has_next() && !jobs_.front()->is_running()) {
        std::unique_ptr<job::Job> job = std::move(jobs_.front());
        jobs_.pop_front();

        unused_job_id_.push_back(job->get_id());

        server_->release_job_instance(std::move(job));
    }

    if (cuda_streams_.empty()) {
        return;
    }

    // TODO: do actual scheduling. Now it is just running whatever runnable, FIFO
    for (const auto& job : jobs_) {
        if (job->has_next() && !job->is_running()) {
            if (job_fits(job.get())) {
                if (!job->has_started()) {
                    server_->notify_job_starts(job.get());
                    job->set_started();
                }
                job->set_running(cuda_streams_.back());
                cuda_streams_.pop_back();
                job::Context::set_current_job(job.get());
                job->run_next();
                break;
            }
        }
    }
}

bool Scheduler::job_fits(job::Job* job) {
    unsigned num_avail_blocks = 0;

    for (auto& sm_avail : sm_avails_) {
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

}
}

