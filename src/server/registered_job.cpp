#include "llis/ipc/shm_channel.h"
#include <llis/server/registered_job.h>

#include <dlfcn.h>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace llis {
namespace server {

RegisteredJob::RegisteredJob(JobRefId registered_job_id, ipc::ShmChannel* c2s_channel, ClientConnection* client_connection) : registered_job_id_(registered_job_id) {
    init(c2s_channel, client_connection);
}

void RegisteredJob::init(ipc::ShmChannel* c2s_channel, ClientConnection* client_connection) {
    c2s_channel_ = c2s_channel;
    client_connection_ = client_connection;

    s2c_channel_ = client_connection_->get_s2c_channel();

    std::string model_path;
    c2s_channel_->read(&model_path);
    void* handle = dlopen(model_path.c_str(), RTLD_LAZY);
    init_job_ = (init_job_t)(dlsym(handle, "init_job"));
    job_ = init_job_();

    c2s_channel_->read(&shm_name_);
    shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0600);

    s2c_channel_->write(registered_job_id_);
}

std::unique_ptr<job::Job> RegisteredJob::create_instance() {
    int mapped_mem_id;
    c2s_channel_->read(&mapped_mem_id);
    size_t offset;
    c2s_channel_->read(&offset);

    std::unique_ptr<job::Job> job(init_job());

    job->full_init(reinterpret_cast<void*>(reinterpret_cast<char*>(mapped_mem_[mapped_mem_id]) + offset));

    return job;
}

void RegisteredJob::grow_pool() {
    void* shm_ptr = mmap(nullptr, pool_size_in_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, pool_size_in_bytes_);
    pool_size_in_bytes_ *= 2;
    mapped_mem_.push_back(shm_ptr);
}

std::unique_ptr<job::Job> RegisteredJob::init_job() {
    std::unique_ptr<job::Job> job(init_job_());
    return job;
}

}
}

