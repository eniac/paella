#include <llis/server/sm_resources.h>

namespace llis {
namespace server {

SmResources::SmResources(int nregs, int smem, int nthrs, int nblocks) : nregs_(nregs), smem_(smem), nthrs_(nthrs), nblocks_(nblocks) {
    max_resources_dot_prod_ = (double)nregs * (double)nregs;
    max_resources_dot_prod_ += (double)nthrs * (double)nthrs;
    max_resources_dot_prod_ += (double)smem * (double)smem;
}

void SmResources::acquire(job::Job* job, int num) {
    // TODO: handle allocation granularity
    nregs_ -= job->get_cur_num_registers_per_thread() * job->get_cur_num_threads_per_block() * num;
    nthrs_ -= job->get_cur_num_threads_per_block() * num;
    smem_ -= job->get_cur_smem_size_per_block() * num;
    nblocks_ -= job->get_cur_num_blocks() * num;
}

void SmResources::release(job::Job* job, int num) {
    // TODO: handle allocation granularity
    nregs_ += job->get_cur_num_registers_per_thread() * job->get_cur_num_threads_per_block() * num;
    nthrs_ += job->get_cur_num_threads_per_block() * num;
    smem_ += job->get_cur_smem_size_per_block() * num;
    nblocks_ += job->get_cur_num_blocks() * num;
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

    unsigned res = nblocks_;

    if (job->get_num_threads_per_block() > 0) {
        res = std::min(res, nregs_ / (job->get_num_registers_per_thread() * job->get_num_threads_per_block()));
        res = std::min(res, nthrs_ / job->get_num_threads_per_block());
    }

    if (job->get_smem_size_per_block() > 0) {
        res = std::min(res, smem_ / job->get_smem_size_per_block());
    }

    return res;
}

float SmResources::normalize_resources(job::Job* job) const {
    return ((float)(job->get_num_registers_per_thread() * job->get_num_threads_per_block()) / nregs_ + (float)job->get_num_threads_per_block() / nthrs_ + (float)job->get_smem_size_per_block() / smem_) * job->get_num_blocks() + (float)job->get_num_blocks() / nblocks_;
}

}
}

