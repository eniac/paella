#pragma once

#include <llis/ipc/shm_channel.h>
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

#ifdef LLIS_MEASURE_BLOCK_TIME
    static void set_gpu2sched_block_time_channel(ipc::Gpu2SchedChannel* gpu2sched_block_time_channel) {
        gpu2sched_block_time_channel_ = gpu2sched_block_time_channel->fork();
    }

    static ipc::Gpu2SchedChannel* get_gpu2sched_block_time_channel() {
        return &gpu2sched_block_time_channel_;
    }
#endif

    static void set_mem2sched_channel(ipc::ShmChannel* mem2sched_channel) {
        mem2sched_channel_ = mem2sched_channel->fork();
    }

    static ipc::ShmChannel* get_mem2sched_channel() {
        return &mem2sched_channel_;
    }

  private:
    static Job* current_job_;
    static ipc::Gpu2SchedChannel gpu2sched_channel_;
#ifdef LLIS_MEASURE_BLOCK_TIME
    static ipc::Gpu2SchedChannel gpu2sched_block_time_channel_;
#endif
    static ipc::ShmChannel mem2sched_channel_;
};

}
}

