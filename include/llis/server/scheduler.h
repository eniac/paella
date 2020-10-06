#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/job.h>

#include <cuda_runtime.h>

#include <deque>
#include <vector>
#include <memory>

namespace llis {
namespace server {

class Scheduer {
  public:
    Scheduer(ipc::ShmChannel* ser2sched_channel);
    void serve();

  private:
    struct SmAvail {
        unsigned nregs = 0;
        unsigned smem = 0;
        unsigned nthrs = 0;
        unsigned nblocks = 0;
    };

    void handle_new_job();
    void handle_block_start_finish();
    void handle_block_start();
    void handle_block_finish();

    void schedule_job();
    bool job_fits(Job* job);

    ipc::ShmChannel* ser2sched_channel_;
    ipc::ShmChannelGpu gpu2sched_channel_;
    
    std::vector<cudaStream_t> cuda_streams_;

    std::deque<std::unique_ptr<Job>> jobs_;

    std::vector<SmAvail> sm_avails_;
};

}
}

