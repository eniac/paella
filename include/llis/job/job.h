#pragma once

#include <llis/ipc/shm_channel.h>

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
    virtual void run_next() = 0;
    virtual bool has_next() const = 0;
    virtual void mark_block_finish() = 0;

    size_t get_output_offset() {
        return (get_input_size() + 8 - 1) & ~(8 - 1);
    }

    size_t get_pinned_mem_size() {
        return get_output_offset() + ((get_output_size() + 8 - 1) & ~(8 - 1));
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
    }

    void set_channel(ipc::ShmChannelGpu&& gpu2sched_channel) {
        gpu2sched_channel_ = std::move(gpu2sched_channel);
    }

    cudaStream_t get_cuda_stream() const {
        return cuda_stream_;
    }

  protected:
    void unset_running() {
        is_running_ = false;
    }

    ipc::ShmChannelGpu gpu2sched_channel_;

  private:
    bool is_running_ = false;
    cudaStream_t cuda_stream_;

    unsigned num_blocks_;
    unsigned num_threads_per_block_;
    unsigned smem_size_per_block_;
    unsigned num_registers_per_thread_;
};

}
}

