#pragma once

#include <llis/job/job.h>

namespace llis {
namespace server {

class SmResources {
  public:
    SmResources(int nregs, int smem, int nthrs, int nblocks);
    SmResources() = default;

    void acquire(job::Job* job, int num);
    void release(job::Job* job, int num);

    double dot(job::Job* job) const;
    double dot_normalized(job::Job* job) const;
    float normalize_resources(job::Job* job) const;

    unsigned num_blocks(job::Job* job) const;

    double occupancy() const;

    bool is_full() const {
        return nregs_ <= 0 || smem_ <= 0 || nthrs_ <= 0 || nblocks_ <= 0;
    }

    bool job_fits(job::Job* job) const {
        return nregs_ >= job->get_num_registers_per_thread() * job->get_num_threads_per_block() && smem_ >= job->get_smem_size_per_block() && nthrs_ >= job->get_num_threads_per_block() && nblocks_ >= job->get_num_blocks();
    }

  private:
    int nregs_ = 0;
    int smem_ = 0;
    int nthrs_ = 0;
    int nblocks_ = 0;

    int max_nregs_ = 0;
    int max_smem_ = 0;
    int max_nthrs_ = 0;
    int max_nblocks_ = 0;

    double max_resources_dot_prod_;
};

}
}

