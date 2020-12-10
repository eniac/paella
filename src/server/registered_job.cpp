#include <llis/ipc/shm_channel.h>
#include <llis/server/registered_job.h>
#include <llis/job/context.h>

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
    void* handle = dlopen(model_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    init_job_ = (init_job_t)(dlsym(handle, "init_job"));
    job_ = init_job_();

    c2s_channel_->read(&shm_name_);
    shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0600);

    pool_size_in_bytes_ = 0;
    mapped_mem_.clear();

    unused_job_instances_.clear();

    stage_lengths_.clear();
    stage_resources_.clear();

    s2c_channel_->write(registered_job_id_);
}

std::unique_ptr<job::Job> RegisteredJob::create_instance() {
    int mapped_mem_id;
    c2s_channel_->read(&mapped_mem_id);
    size_t offset;
    c2s_channel_->read(&offset);
    void* remote_ptr;
    c2s_channel_->read(&remote_ptr);

    if (unused_job_instances_.empty()) {
        std::unique_ptr<job::Job> job(init_job());

        job->set_client_details(client_connection_->get_client_id(), registered_job_id_);
        job->set_remote_ptr(remote_ptr);
        job::Context::set_current_job(job.get());
        job->full_init(reinterpret_cast<void*>(reinterpret_cast<char*>(mapped_mem_[mapped_mem_id]) + offset));

        return job;
    } else {
        std::unique_ptr<job::Job> job = std::move(unused_job_instances_.back());
        unused_job_instances_.pop_back();

        job->set_remote_ptr(remote_ptr);
        job->reset_internal();
        job::Context::set_current_job(job.get());
        job->init(reinterpret_cast<void*>(reinterpret_cast<char*>(mapped_mem_[mapped_mem_id]) + offset));

        return job;
    }
}

void RegisteredJob::release_instance(std::unique_ptr<job::Job> job) {
    unused_job_instances_.push_back(std::move(job));
}

void RegisteredJob::grow_pool() {
    size_t num_new_bytes;
    c2s_channel_->read(&num_new_bytes);

    void* shm_ptr = mmap(nullptr, num_new_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, pool_size_in_bytes_);
    cudaHostRegister(shm_ptr, num_new_bytes, cudaHostRegisterDefault);
    pool_size_in_bytes_ += num_new_bytes;
    mapped_mem_.push_back(shm_ptr);
}

std::unique_ptr<job::Job> RegisteredJob::init_job() {
    std::unique_ptr<job::Job> job(init_job_());
    return job;
}

void RegisteredJob::update_stage_length(unsigned stage_id, double len) {
    if (stage_lengths_.size() > stage_id) {
        // TODO: tune the update rule
        stage_lengths_[stage_id] = (stage_lengths_[stage_id] + len) / 2;
    } else {
        stage_lengths_.push_back(len);
    }
}

void RegisteredJob::set_stage_resource(unsigned stage_id, float res) {
    if (stage_resources_.size() <= stage_id) {
        stage_resources_.push_back(res);
    }
    // Don't need to do anything otherwise, because resource needs do not change
}

double RegisteredJob::get_stage_length(unsigned stage_id) const {
    if (stage_lengths_.size() > stage_id) {
        return stage_lengths_[stage_id];
    } else {
        // TODO: tune the default value when it is unknown
        return 0;
    }
}

float RegisteredJob::get_stage_resource(unsigned stage_id) const {
    if (stage_resources_.size() > stage_id) {
        return stage_resources_[stage_id];
    } else {
        // TODO: tune the default value when it is unknown
        return 0;
    }
}

double RegisteredJob::get_remaining_length(unsigned from_stage) const {
    double sum = 0;
    for (unsigned i = from_stage; i < stage_lengths_.size(); ++i) {
        sum += stage_lengths_[i];
    }
    return sum;
}

double RegisteredJob::get_remaining_rl(unsigned from_stage) const {
    double sum = 0;
    for (unsigned i = from_stage; i < stage_lengths_.size(); ++i) {
        sum += stage_lengths_[i] * stage_resources_[i];
    }
    return sum;
}

}
}

