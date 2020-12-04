#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/job/job.h>
#include <llis/server/server.h>

#include <cuda_runtime.h>

#include <deque>
#include <vector>
#include <memory>

namespace llis {
namespace server {

class Scheduler {
  public:
    Scheduler();

    void set_server(Server* server);

    void handle_new_job(std::unique_ptr<job::Job> job);
    void try_handle_block_start_finish();

  private:
    struct SmAvail {
        unsigned nregs = 0;
        unsigned smem = 0;
        unsigned nthrs = 0;
        unsigned nblocks = 0;
    };

    void handle_block_start_finish();
    void handle_block_start();
    void handle_block_finish();

    void schedule_job();
    bool job_fits(job::Job* job);

    Server* server_;
    ipc::ShmChannelGpu gpu2sched_channel_;
    
    std::vector<cudaStream_t> cuda_streams_;

    std::deque<std::unique_ptr<job::Job>> jobs_;

    std::vector<SmAvail> sm_avails_;
};

}
}

