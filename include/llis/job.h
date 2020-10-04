#pragma once

#include <llis/ipc/shm_channel.h>

#include <cstddef>

namespace llis {

class Job {
  public:
    virtual size_t get_input_size() = 0;
    virtual size_t get_output_size() = 0;
    virtual size_t get_param_size() = 0;
    virtual void full_init(void* io_ptr) = 0;
    virtual void run_next() = 0;
    virtual bool has_next() const = 0;
    virtual void mark_block_finish() = 0;
    
    bool is_running() const {
        return is_running_;
    }

    void set_running() {
        is_running_ = true;
    }

    void set_channel(ipc::ShmChannelGpu&& gpu2sched_channel) {
        gpu2sched_channel_ = std::move(gpu2sched_channel);
    }

  protected:
    void unset_running() {
        is_running_ = false;
    }

    ipc::ShmChannelGpu gpu2sched_channel_;

  private:
    bool is_running_ = false;
};

}
