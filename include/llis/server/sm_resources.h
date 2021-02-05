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

    bool is_full() const {
        return nregs_ <= 0 || smem_ <= 0 || nthrs_ <= 0 || nblocks_ <= 0;
    }

  private:
    int nregs_ = 0;
    int smem_ = 0;
    int nthrs_ = 0;
    int nblocks_ = 0;

    double max_resources_dot_prod_;
};

}
}

