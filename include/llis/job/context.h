#pragma once

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/job/job.h>

namespace llis {
namespace job {

class Context {
  public:
    static Job* get_current_job() {
        return current_job_;
    }

    static void set_current_job(Job* job) {
        current_job_ = job;
    }

    static void set_gpu2sched_channel(ipc::Gpu2SchedChannel* gpu2sched_channel) {
        gpu2sched_channel_ = gpu2sched_channel->fork();
    }

    static ipc::Gpu2SchedChannel* get_gpu2sched_channel() {
        return &gpu2sched_channel_;
    }

  private:
    static Job* current_job_;
    static ipc::Gpu2SchedChannel gpu2sched_channel_;
};

}
}

