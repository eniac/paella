#include <llis/ipc/shm_channel.h>
#include <llis/server/client_connection.h>
#include <llis/ipc/defs.h>
#include <llis/server/registered_job.h>

#include <vector>
#include <string>

namespace llis {
namespace server {

const size_t s2c_channel_size = 1024;

class Server {
  public:
    Server(std::string server_name, ipc::ShmChannel* ser2sched_channel);

    void serve();

  private:
    void handle_register_client();
    void handle_register_job();
    void handle_launch_job();
    void handle_grow_pool();
    void handle_release_job_instance();

    std::string server_name_;
    ipc::ShmChannel* ser2sched_channel_;

    ipc::ShmChannel c2s_channel_;

    std::vector<ClientConnection> client_connections_;
    std::vector<ClientId> unused_client_connections_;

    std::vector<RegisteredJob> registered_jobs_;
    std::vector<JobRefId> unused_registered_jobs_;
};

}
}

