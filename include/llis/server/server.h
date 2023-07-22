#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/ipc/unix_datagram_socket.h>
#include <llis/server/client_connection.h>
#include <llis/ipc/defs.h>
#include <llis/server/registered_job.h>
#include <llis/server/profiler.h>
#include <llis/utils/logging.hh>

#include <vector>
#include <string>

#define SER2SCHED_CHAN_SIZE 1024
#define CLT2SCHED_CHAN_SIZE 1024

namespace llis {
namespace server {

class Scheduler;

constexpr size_t s2c_channel_size = 4096;

class Server {
  public:
    Server(std::string server_name, Scheduler* scheduler);

    void serve();

    void notify_job_starts(job::Job* job);
    void notify_job_ends(job::Job* job);

    void release_job_instance(std::unique_ptr<job::Job> job);

    void update_job_stage_length(job::Job* job, unsigned stage_id, double len);
    void set_job_stage_resource(job::Job* job, unsigned stage_id, float res);
    bool has_job_stage_resource(job::Job* job, unsigned stage_id);

    double get_job_remaining_length(job::Job* job, unsigned from_stage) const;
    double get_job_remaining_rl(job::Job* job, unsigned from_stage) const;

    const std::vector<double>& get_job_stage_lengths(job::Job* job) const;
    const std::vector<float>& get_job_stage_resources(job::Job* job) const;

    Profiler* get_profiler() {
        return &profiler_;
    }

  private:
    void try_handle_c2s();
    void handle_c2s();
    void handle_register_client();
    void handle_register_job();
    void handle_launch_job();
    void handle_grow_pool();
    void handle_release_job_instance();

    std::string server_name_;
    Scheduler* scheduler_;
    ipc::UnixDatagramSocket s2c_socket_;

    ipc::ShmChannelCpuReader c2s_channel_;

    std::vector<ClientConnection> client_connections_;
    std::vector<ClientId> unused_client_connections_;

    std::vector<RegisteredJob> registered_jobs_;
    std::vector<JobRefId> unused_registered_jobs_;

    Profiler profiler_;
};

}
}

