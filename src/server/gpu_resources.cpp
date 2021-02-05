#include <llis/job/job.h>
#include <llis/server/gpu_resources.h>

#include <algorithm>

namespace llis {
namespace server {

GpuResources::GpuResources() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // TODO: handle multiple GPUs

    int num_sms = prop.multiProcessorCount;

    int total_nregs = 0;
    int total_smem = 0;
    int total_nthrs = 0;
    int total_nblocks = 0;

    // TODO: Some of them can be zero because some smid may not reflect an actual SM
    sms_resources_.reserve(num_sms);

    for (int i = 0; i < num_sms; ++i) {
        sms_resources_.emplace_back(prop.regsPerMultiprocessor, prop.sharedMemPerMultiprocessor, prop.maxThreadsPerMultiProcessor, prop.maxBlocksPerMultiProcessor);

        total_nregs += prop.regsPerMultiprocessor;
        total_smem += prop.sharedMemPerMultiprocessor;
        total_nthrs += prop.maxThreadsPerMultiProcessor;
        total_nblocks += prop.maxBlocksPerMultiProcessor;
    }

    total_resources_ = SmResources(total_nregs, total_smem, total_nthrs, total_nblocks);

    gpc_num_blocks_.resize(5);
    gpc_next_sms_.resize(5);
}

void GpuResources::acquire(int smid, job::Job* job, int num) {
    SmResources* sm_resources = &sms_resources_[smid];

    bool before_is_full = sm_resources->is_full();
    sm_resources->acquire(job, num);
    bool after_is_full = sm_resources->is_full();

    if (!before_is_full && after_is_full) {
        ++num_full_sms_;
    }

    total_resources_.acquire(job, num);
}

void GpuResources::release(int smid, job::Job* job, int num) {
    SmResources* sm_resources = &sms_resources_[smid];

    bool before_is_full = sm_resources->is_full();
    sm_resources->release(job, num);
    bool after_is_full = sm_resources->is_full();

    if (before_is_full && !after_is_full) {
        --num_full_sms_;
    }

    total_resources_.release(job, num);
}

bool GpuResources::job_fits(job::Job* job) const {
    unsigned num_avail_blocks = 0;

    for (const SmResources& sm_resources : sms_resources_) {
        num_avail_blocks += sm_resources.num_blocks(job);

        if (num_avail_blocks >= job->get_num_blocks()) {
            return true;
        }
    }

    return false;
}

void GpuResources::choose_sms(job::Job* job) {
    unsigned num_blocks = job->get_num_blocks();

    for (unsigned blockId = 0; blockId < num_blocks; ++blockId) {
        unsigned gpc = std::min_element(gpc_num_blocks_.begin(), gpc_num_blocks_.end()) - gpc_num_blocks_.begin();
        unsigned smid = gpc_sms_[gpc][gpc_next_sms_[gpc]];

        ++gpc_num_blocks_[gpc];
        gpc_next_sms_[gpc] = (gpc_next_sms_[gpc] + 1) % 8;

        acquire(smid, job, 1);

        // TODO: handle overusing resources

        job->add_predicted_smid(smid);
    }
}

double GpuResources::dot(job::Job* job) const {
    return total_resources_.dot(job);
}

double GpuResources::dot_normalized(job::Job* job) const {
    return total_resources_.dot_normalized(job);
}

float GpuResources::normalize_resources(job::Job* job) const {
    return total_resources_.normalize_resources(job);
}

}
}

