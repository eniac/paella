#include <llis/ipc/shm_channel.h>
#include <llis/server/scheduler.h>
#include <llis/job.h>
#include <memory>
#include <string>

namespace llis {
namespace server {

Scheduer::Scheduer(ipc::ShmChannel* ser2sched_channel) : ser2sched_channel_(ser2sched_channel), gpu2sched_channel_(1024) {
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
    // TODO
}

void Scheduer::handle_block_finish() {
    Job* job;
    gpu2sched_channel_.read(&job);

    job->mark_block_finish();

    // TODO: handle resouce release

    schedule_job();
}

void Scheduer::handle_new_job() {
    std::unique_ptr<Job> job;
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
            job->set_running();
            job->run_next();
            break;
        }
    }
}

}
}

