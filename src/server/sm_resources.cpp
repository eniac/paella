#include <llis/server/sm_resources.h>

namespace llis {
namespace server {

SmResources::SmResources(int nregs, int smem, int nthrs, int nblocks) : nregs_(nregs), smem_(smem), nthrs_(nthrs), nblocks_(nblocks) {
    max_resources_dot_prod_ = (double)nregs * (double)nregs;
    max_resources_dot_prod_ += (double)nthrs * (double)nthrs;
    max_resources_dot_prod_ += (double)smem * (double)smem;

    max_nregs_ = nregs;
    max_smem_ = smem;
    max_nthrs_ = nthrs;
    max_nblocks_ = nblocks;
}

SmResources::SmResources() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // TODO: handle multiple GPUs

    nregs_ = prop.regsPerMultiprocessor * prop.multiProcessorCount;
    smem_ = prop.sharedMemPerMultiprocessor * prop.multiProcessorCount;
    nthrs_ = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    nblocks_ = prop.maxBlocksPerMultiProcessor * prop.multiProcessorCount;

    max_resources_dot_prod_ = (double)nregs_ * (double)nregs_;
    max_resources_dot_prod_ += (double)nthrs_ * (double)nthrs_;
    max_resources_dot_prod_ += (double)smem_ * (double)smem_;

    max_nregs_ = nregs_;
    max_smem_ = smem_;
    max_nthrs_ = nthrs_;
    max_nblocks_ = nblocks_;
}

void SmResources::acquire(job::Job* job, int num) {
    // TODO: handle allocation granularity
    nregs_ -= job->get_cur_num_registers_per_thread() * job->get_cur_num_threads_per_block() * num;
    nthrs_ -= job->get_cur_num_threads_per_block() * num;
    smem_ -= job->get_cur_smem_size_per_block() * num;
    nblocks_ -= num;

#ifdef PRINT_RESOURCES
    printf("Resources: %d %d %d %d\n", nregs_, nthrs_, smem_, nblocks_);
#endif

#ifdef LLIS_ENABLE_PROFILER
    profiler_->record_resource_event(job, num, Profiler::ResourceEvent::ACQUIRE);
#endif
}

void SmResources::release(job::Job* job, int num) {
    // TODO: handle allocation granularity
    nregs_ += job->get_cur_num_registers_per_thread() * job->get_cur_num_threads_per_block() * num;
    nthrs_ += job->get_cur_num_threads_per_block() * num;
    smem_ += job->get_cur_smem_size_per_block() * num;
    nblocks_ += num;

#ifdef PRINT_RESOURCES
    printf("Resources: %d %d %d %d\n", nregs_, nthrs_, smem_, nblocks_);
#endif

#ifdef LLIS_ENABLE_PROFILER
    profiler_->record_resource_event(job, num, Profiler::ResourceEvent::RELEASE);
#endif
}

double SmResources::dot(job::Job* job) const {
    double res = (double)nregs_ * (double)job->get_cur_num_registers_per_thread() * (double)job->get_cur_num_threads_per_block();
    res += (double)nthrs_ * (double)job->get_cur_num_threads_per_block();
    res += (double)smem_ * (double)job->get_smem_size_per_block();
    return res;
}

double SmResources::dot_normalized(job::Job* job) const {
    return dot(job) / max_resources_dot_prod_;
}

unsigned SmResources::num_blocks(job::Job* job) const {
    if (is_full()) {
        return 0;
    }

    int res = nblocks_;

    if (job->get_num_threads_per_block() > 0) {
        res = std::min(res, nregs_ / (int)(job->get_num_registers_per_thread() * job->get_num_threads_per_block()));
        res = std::min(res, nthrs_ / (int)job->get_num_threads_per_block());
    }

    if (job->get_smem_size_per_block() > 0) {
        res = std::min(res, smem_ / (int)job->get_smem_size_per_block());
    }

    if (res > 0) {
        return res;
    } else {
        return 0;
    }
}

float SmResources::normalize_resources(job::Job* job) const {
    return ((float)(job->get_num_registers_per_thread() * job->get_num_threads_per_block()) / max_nregs_ + (float)job->get_num_threads_per_block() / max_nthrs_ + (float)job->get_smem_size_per_block() / max_smem_) * job->get_num_blocks() + (float)job->get_num_blocks() / max_nblocks_;
}

double SmResources::occupancy() const {
    double a = 1. - (double)nthrs_ / (double)max_nthrs_;
    double b = 1. - (double)smem_ / (double)max_smem_;
    double c = 1. - (double)nregs_ / (double)max_nregs_;
    return std::max(a, std::max(b, c));
}

}
}

