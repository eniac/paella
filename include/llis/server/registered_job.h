#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/server/client_connection.h>
#include <llis/job.h>

#include <vector>

namespace llis {
namespace server {

class RegisteredJob {
  public:
    RegisteredJob(JobRefId registered_job_id, ipc::ShmChannel* c2s_channel_, ClientConnection* client_connection);

    void init(ipc::ShmChannel* c2s_channel, ClientConnection* client_connection);

    void launch();
    void grow_pool();

  private:
    JobRefId registered_job_id_;
    ipc::ShmChannel* c2s_channel_;
    ClientConnection* client_connection_;

    ipc::ShmChannel* s2c_channel_;
    Job* job_;
    std::string shm_name_;
    int shm_fd_;

    size_t pool_size_in_bytes_;
    std::vector<void*> mapped_mem_;
};

}
}

