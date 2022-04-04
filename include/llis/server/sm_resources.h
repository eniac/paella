#pragma once

#include <llis/job/job.h>
#include <llis/server/profiler.h>

namespace llis {
namespace server {

class SmResources {
  public:
    SmResources(int nregs, int smem, int nthrs, int nblocks);
    SmResources();

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
        return nregs_ >= (int)job->get_num_registers_per_thread() * (int)job->get_num_threads_per_block() * (int)job->get_num_blocks() && smem_ >= (int)job->get_smem_size_per_block() * (int)job->get_num_blocks() && nthrs_ >= (int)job->get_num_threads_per_block() * (int)job->get_num_blocks() && nblocks_ >= (int)job->get_num_blocks();
    }

#ifdef LLIS_ENABLE_PROFILER
    void set_profiler(Profiler* profiler) {
        profiler_ = profiler;
    }
#endif

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

    Profiler* profiler_;
};

}
}

