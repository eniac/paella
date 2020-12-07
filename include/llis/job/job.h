#pragma once

#include <llis/utils/align.h>
#include <llis/ipc/defs.h>

#include <cuda_runtime.h>

#include <cstddef>

namespace llis {
namespace job {

class Job {
  public:
    virtual size_t get_input_size() = 0;
    virtual size_t get_output_size() = 0;
    virtual size_t get_param_size() = 0;
    virtual void full_init(void* io_ptr) = 0;
    virtual void init(void* io_ptr) = 0;
    virtual void run_next() = 0;
    virtual bool has_next() const = 0;

    void set_id(JobId id) {
        id_ = id;
    }

    JobId get_id() const {
        return id_;
    }

    size_t get_output_offset() {
        return utils::next_aligned_pos(get_input_size(), 8);
    }

    size_t get_pinned_mem_size() {
        return get_output_offset() + utils::next_aligned_pos(get_output_size(), 8);
    }

    unsigned get_num_blocks() const {
        return num_blocks_;
    }

    unsigned get_num_threads_per_block() const {
        return num_threads_per_block_;
    }

    unsigned get_smem_size_per_block() const {
        return smem_size_per_block_;
    }

    unsigned get_num_registers_per_thread() const {
        return num_registers_per_thread_;
    }

    void set_num_blocks(unsigned num_blocks) {
        num_blocks_ = num_blocks;
    }

    void set_num_threads_per_block(unsigned num_threads_per_block) {
        num_threads_per_block_ = num_threads_per_block;
    }

    void set_smem_size_per_block(unsigned smem_size_per_block) {
        smem_size_per_block_ = smem_size_per_block;
    }

    void set_num_registers_per_thread(unsigned num_registers_per_thread) {
        num_registers_per_thread_ = num_registers_per_thread;
    }

    bool is_running() const {
        return is_running_;
    }

    void set_running(cudaStream_t cuda_stream) {
        is_running_ = true;
        cuda_stream_ = cuda_stream;
        num_running_blocks_ = num_blocks_;
    }

    void unset_running() {
        is_running_ = false;
    }

    void mark_block_finish() {
        num_running_blocks_--;
        if (num_running_blocks_ == 0) {
            unset_running();
        }
    }

    cudaStream_t get_cuda_stream() const {
        return cuda_stream_;
    }

    void set_started() {
        has_started_ = true;
    }

    void unset_started() {
        has_started_ = false;
    }

    bool has_started() const {
        return has_started_;
    }

    void set_client_details(ClientId client_id, JobRefId registered_job_id) {
        client_id_ = client_id;
        registered_job_id_ = registered_job_id;
    }

    ClientId get_client_id() const {
        return client_id_;
    }

    JobRefId get_registered_job_id() const {
        return registered_job_id_;
    }

    JobInstanceRefId get_job_instance_ref_id() const {
        return job_instance_ref_id_;
    }

    void set_job_instance_ref_id(JobInstanceRefId id) {
        job_instance_ref_id_ = id;
    }

  private:
    bool is_running_ = false;
    cudaStream_t cuda_stream_;

    unsigned num_blocks_;
    unsigned num_threads_per_block_;
    unsigned smem_size_per_block_;
    unsigned num_registers_per_thread_;

    unsigned num_running_blocks_;

    bool has_started_ = false;
    ClientId client_id_;
    JobRefId registered_job_id_;
    JobInstanceRefId job_instance_ref_id_;
    JobId id_;
};

}
}

