#pragma once

#include <llis/ipc/shm_channel.h>
#include <llis/server/client_connection.h>
#include <llis/job/job.h>

#include <vector>
#include <memory>

namespace llis {
namespace server {

class RegisteredJob {
  public:
    RegisteredJob(JobRefId registered_job_id, ipc::ShmChannel* c2s_channel_, ClientConnection* client_connection);
    RegisteredJob(RegisteredJob&&) = default;

    void init(ipc::ShmChannel* c2s_channel, ClientConnection* client_connection);

    std::unique_ptr<job::Job> create_instance();
    void grow_pool();
    std::unique_ptr<job::Job> init_job();
    void release_instance(std::unique_ptr<job::Job> job);

    void update_stage_length(unsigned stage_id, double len);
    void set_stage_resource(unsigned stage_id, float res);
    double get_stage_length(unsigned stage_id) const;
    float get_stage_resource(unsigned stage_id) const;
    double get_remaining_length(unsigned from_stage) const;
    double get_remaining_rl(unsigned from_stage) const;

  private:
    typedef job::Job* (*init_job_t)();

    JobRefId registered_job_id_;
    ipc::ShmChannel* c2s_channel_;
    ClientConnection* client_connection_;

    ipc::ShmChannel* s2c_channel_;
    init_job_t init_job_;
    job::Job* job_;
    std::string shm_name_;
    int shm_fd_;

    size_t pool_size_in_bytes_;
    std::vector<void*> mapped_mem_;

    std::vector<std::unique_ptr<job::Job>> unused_job_instances_;

    std::vector<double> stage_lengths_;
#ifdef PRINT_STAGE_LENGTH_STDDEV
    std::vector<double> stage_lengths_sum_;
    std::vector<double> stage_lengths_sum_sqr_;
    std::vector<unsigned long long> stage_lengths_num_;
#endif
    std::vector<float> stage_resources_;
};

}
}

