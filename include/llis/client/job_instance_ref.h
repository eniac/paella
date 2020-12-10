#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/ipc/defs.h>
#include <llis/client/io_shm_entry.h>

#include <chrono>

namespace llis {
namespace client {

class JobRef;

class JobInstanceRef {
  public:
    JobInstanceRef(JobRef* job_ref, IoShmEntry io_shm_entry);
    ~JobInstanceRef();

    void launch();
    void release();

    void* get_input_ptr();
    void* get_output_ptr();

    void set_id(JobInstanceRefId id);
    JobInstanceRefId get_id() const;

    void record_start_time();
    void set_start_time(std::chrono::time_point<std::chrono::steady_clock> time_point);
    std::chrono::time_point<std::chrono::steady_clock> get_start_time() const;

  private:
    JobRef* job_ref_;
    IoShmEntry io_shm_entry_;

    JobInstanceRefId id_;

    ipc::ShmChannel* c2s_channel_;

    std::chrono::time_point<std::chrono::steady_clock> start_time_;
};

}
}

