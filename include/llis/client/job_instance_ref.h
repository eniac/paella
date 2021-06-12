#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/ipc/defs.h>
#include <llis/client/io_shm_entry.h>

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

    JobRefId get_job_ref_id() const;

    void set_start_time(double time_point);
    double get_start_time() const;

  private:
    JobRef* job_ref_;
    IoShmEntry io_shm_entry_;

    JobInstanceRefId id_;

    ipc::ShmChannelCpuWriter* c2s_channel_;

    double start_time_;
};

}
}

