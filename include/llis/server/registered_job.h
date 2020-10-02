#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/server/client_connection.h>
#include <llis/job.h>

#include <vector>
#include <memory>

namespace llis {
namespace server {

class RegisteredJob {
  public:
    RegisteredJob(JobRefId registered_job_id, ipc::ShmChannel* c2s_channel_, ClientConnection* client_connection);
    RegisteredJob(RegisteredJob&&) = default;

    void init(ipc::ShmChannel* c2s_channel, ClientConnection* client_connection);

    std::unique_ptr<Job> create_instance();
    void grow_pool();
    std::unique_ptr<Job> init_job();

  private:
    typedef Job* (*init_job_t)();

    JobRefId registered_job_id_;
    ipc::ShmChannel* c2s_channel_;
    ClientConnection* client_connection_;

    ipc::ShmChannel* s2c_channel_;
    init_job_t init_job_;
    Job* job_;
    std::string shm_name_;
    int shm_fd_;

    size_t pool_size_in_bytes_;
    std::vector<void*> mapped_mem_;
};

}
}

