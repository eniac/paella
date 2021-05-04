#include <llis/client/client.h>
#include <llis/job/job.h>
#include <llis/ipc/name_format.h>

#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <random>
#include <string>

namespace llis {
namespace client {

Client::Client(std::string server_name) :
        server_name_(std::move(server_name)),
        c2s_channel_(ipc::c2s_channel_name(server_name_)),
        profiler_client_(&c2s_channel_) {
    generate_client_id();
    create_s2c_channel();
    register_client();
    reconnect_s2c_channel(); // reconnect after getting the permanent client id
    connect_s2c_socket();
}

Client::~Client() {
    // TODO
}

void Client::generate_client_id() {
    std::random_device rnd;
    client_id_ = rnd();
}

void Client::create_s2c_channel() {
    s2c_channel_.connect(ipc::s2c_channel_name(server_name_, client_id_), 8);
}

void Client::reconnect_s2c_channel() {
    s2c_channel_.disconnect();
    s2c_channel_.connect(ipc::s2c_channel_name(server_name_, client_id_));
}

void Client::connect_s2c_socket() {
    s2c_socket_.bind(ipc::s2c_socket_name(server_name_, client_id_));
}

void Client::register_client() {
    c2s_channel_.acquire_writer_lock();

    c2s_channel_.write(MsgType::REGISTER_CLIENT);
    c2s_channel_.write(client_id_);

    c2s_channel_.release_writer_lock();

    // The random client id was only temporary. The server will assign a permanent client id.
    s2c_channel_.read(&client_id_);
}

JobRef Client::register_job(std::string path) {
    void* handle = dlopen(path.c_str(), RTLD_NOW);
    if (handle == NULL) {
        printf("Failed to read job definition: %s\n", dlerror());
    }
    typedef job::Job* (*init_job_t)();
    init_job_t init_job = (init_job_t)(dlsym(handle, "init_job"));
    // TODO: error handling

    job::Job* job = init_job();
    JobRef job_ref(job, this, path);

    return job_ref;
}

JobInstanceRef* Client::add_job_instance_ref(JobInstanceRef job_instance_ref) {
    std::unique_lock<std::mutex> lk(mtx_);

    if (unused_job_instance_refs_.empty()) {
        lk.unlock();

        JobInstanceRefId id = job_instance_refs_.size();
        job_instance_ref.set_id(id);
        job_instance_refs_.push_back(std::make_unique<JobInstanceRef>(std::move(job_instance_ref)));
        return job_instance_refs_.back().get();
    } else {
        JobInstanceRefId id = unused_job_instance_refs_.back();
        unused_job_instance_refs_.pop_back();

        lk.unlock();

        job_instance_ref.set_id(id);
        *(job_instance_refs_[id]) = std::move(job_instance_ref);
        return job_instance_refs_[id].get();
    }
}

void Client::release_job_instance_ref(JobInstanceRef* job_instance_ref) {
    std::lock_guard<std::mutex> lk(mtx_);

    unused_job_instance_refs_.push_back(job_instance_ref->get_id());
}

JobInstanceRef* Client::wait() {
    // Wait for start notification
    bool flag;
    s2c_socket_.read(&flag, sizeof(flag));

    // Wait for end notification
    JobInstanceRef* res;
    s2c_channel_.read(&res);

    return res;
}

void Client::kill_server() {
    c2s_channel_.acquire_writer_lock();

    c2s_channel_.write(MsgType::EXIT_CMD);

    c2s_channel_.release_writer_lock();
}

}
}

