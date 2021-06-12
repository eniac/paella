#pragma once

#include <llis/client/job_instance_ref.h>
#include <llis/job/job.h>
#include <llis/ipc/shm_channel.h>
#include <llis/ipc/defs.h>

#include <vector>
#include <string>

namespace llis {
namespace client {

class Client;

class JobRef {
  public:
    JobRef(std::unique_ptr<job::Job> job, Client* client, std::string path);
    ~JobRef();

    JobRef(const JobRef&) = delete;
    JobRef(JobRef&&) = default;
    JobRef& operator=(const JobRef&) = delete;
    JobRef& operator=(JobRef&&) = default;

    JobInstanceRef* create_instance();
    void release_io_shm_entry(IoShmEntry io_shm_entry);

    job::Job* get_job() {
        return job_.get();
    }

    Client* get_client() {
        return client_;
    }

    ClientId get_client_id() const {
        return client_id_;
    }

    JobRefId get_job_ref_id() const {
        return job_ref_id_;
    }

    ipc::ShmChannelCpuReader* get_s2c_channel() {
        return s2c_channel_;
    }

    ipc::ShmChannelCpuWriter* get_c2s_channel() {
        return c2s_channel_;
    }

  private:
    void register_job();

    void grow_pool(size_t least_num_new_entries);
    void grow_pool();

    std::unique_ptr<job::Job> job_;
    Client* client_;
    std::string model_path_;

    ipc::ShmChannelCpuReader* s2c_channel_;
    ipc::ShmChannelCpuWriter* c2s_channel_;
    ClientId client_id_;

    size_t pinned_mem_size_;
    size_t param_size_;

    size_t pool_size_ = 0; // number of concurrent instances that can be supported
    size_t pool_size_in_bytes_ = 0; // number of bytes of the pool

    std::vector<void*> pinned_mem_list_;
    std::vector<IoShmEntry> pinned_mem_free_list_;

    std::vector<void*> param_mem_list_;
    std::vector<IoShmEntry> param_mem_free_list_;

    std::string shm_name_;
    int shm_fd_;

    JobRefId job_ref_id_;
};

}
}

