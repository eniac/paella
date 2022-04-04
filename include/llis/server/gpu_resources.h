#pragma once

#include <llis/job/job.h>
#include <llis/server/sm_resources.h>
#include <llis/server/profiler.h>

namespace llis {
namespace server {

class GpuResources {
  public:
    GpuResources();

    void acquire(int smid, job::Job* job, int num);
    void release(int smid, job::Job* job, int num);

    bool job_fits(job::Job* job) const;

    unsigned get_num_sms() const {
        return sms_resources_.size();
    }

    bool is_full() const {
        return num_full_sms_ >= sms_resources_.size();
    }

    void choose_sms(job::Job* job);

    double dot(job::Job* job) const;
    double dot_normalized(job::Job* job) const;
    float normalize_resources(job::Job* job) const;

#ifdef LLIS_ENABLE_PROFILER
    void set_profiler(Profiler* profiler) {
        for (auto& sm_resources : sms_resources_) {
            sm_resources.set_profiler(profiler);
        }
    }
#endif

  private:
    std::vector<SmResources> sms_resources_;
    SmResources total_resources_;
    unsigned num_full_sms_ = 0;

    std::vector<unsigned> gpc_num_blocks_;
    std::vector<unsigned> gpc_next_sms_;
    // TODO: detect the actual allocation
    constexpr static unsigned gpc_sms_[5][8] = {{0, 10, 20, 30, 1, 11, 21, 31}, {2, 12, 22, 32, 3, 13, 23, 33}, {4, 14, 24, 34, 5, 15, 25, 35}, {6, 16, 26, 36, 7, 17, 27, 37}, {8, 18, 28, 38, 9, 19, 29, 39}};
};

}
}

