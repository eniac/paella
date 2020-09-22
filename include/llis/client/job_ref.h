#pragma once

#include "job_instance_ref.h"
#include <llis/job.h>
#include <llis/ipc/shm_channel.h>
#include <llis/ipc/defs.h>

#include <vector>
#include <string>

namespace llis {

class Client;

class JobRef {
  public:
    JobRef(Job* job, Client* client, std::string path);

    JobInstanceRef create_instance();

    Job* get_job() {
        return job_;
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

    ipc::ShmChannel* get_s2c_channel() {
        return s2c_channel_;
    }

    ipc::ShmChannel* get_c2s_channel() {
        return c2s_channel_;
    }

  private:
    void register_job();

    void grow_pool();

    Job* job_;
    Client* client_;
    std::string model_path_;

    ipc::ShmChannel* s2c_channel_;
    ipc::ShmChannel* c2s_channel_;
    ClientId client_id_;

    size_t input_size_;
    size_t output_size_;
    size_t pinned_mem_size_;
    size_t param_size_;

    size_t pool_size_; // number of concurrent instances that can be supported
    size_t pool_size_in_bytes_; // number of bytes of the pool

    std::vector<void*> pinned_mem_list_;
    std::vector<IoShmEntry> pinned_mem_free_list_;

    std::vector<void*> param_mem_list_;
    std::vector<IoShmEntry> param_mem_free_list_;

    std::string shm_name_;
    int shm_fd_;

    JobRefId job_ref_id_;
};

}

