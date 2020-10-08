#include <llis/ipc/shm_channel.h>
#include <llis/server/scheduler.h>
#include <llis/job/job.h>
#include <memory>
#include <string>

namespace llis {
namespace server {

Scheduer::Scheduer(ipc::ShmChannel* ser2sched_channel) : ser2sched_channel_(ser2sched_channel), gpu2sched_channel_(1024), cuda_streams_(100) {
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

void Scheduer::serve() {
    while (true) {
        if (ser2sched_channel_->can_read()) {
            handle_new_job();
        }

        if (gpu2sched_channel_.can_read()) {
            handle_block_start_finish();
        }
    }
}

void Scheduer::handle_block_start_finish() {
    bool is_start;
    gpu2sched_channel_.read(&is_start);
    
    if (is_start) {
        handle_block_start();
    } else {
        handle_block_finish();
    }
}

void Scheduer::handle_block_start() {
    job::Job* job;
    gpu2sched_channel_.read(&job);

    unsigned smid;
    gpu2sched_channel_.read(&smid);

    // TODO: handle allocation granularity
    sm_avails_[smid].nregs -= job->get_num_registers_per_thread() * job->get_num_threads_per_block(); // TODO: use an actual number
    sm_avails_[smid].nthrs -= job->get_num_threads_per_block();
    sm_avails_[smid].smem -= job->get_smem_size_per_block();
}

void Scheduer::handle_block_finish() {
    job::Job* job;
    gpu2sched_channel_.read(&job);

    job->mark_block_finish();
    if (!job->is_running()) {
        cuda_streams_.push_back(job->get_cuda_stream());
    }

    // TODO: handle resouce release

    schedule_job();
}

void Scheduer::handle_new_job() {
    std::unique_ptr<job::Job> job;
    ser2sched_channel_->read(&job);

    job->set_channel(gpu2sched_channel_.fork());

    jobs_.push_back(std::move(job));

    schedule_job();
}

void Scheduer::schedule_job() {
    while (!jobs_.empty() && !jobs_.front()->has_next() && !jobs_.front()->is_running()) {
        jobs_.pop_front();
    }

    // TODO: do actual scheduling. Now it is just running whatever runnable, FIFO
    for (const auto& job : jobs_) {
        if (job->has_next() && !job->is_running()) {
            if (job_fits(job.get())) {
                job->set_running(cuda_streams_.back());
                cuda_streams_.pop_back();
                job->run_next();
                break;
            }
        }
    }
}

bool Scheduer::job_fits(job::Job* job) {
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

