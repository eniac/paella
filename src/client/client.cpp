#include <llis/client/client.h>
#include <llis/job/job.h>
#include <llis/ipc/name_format.h>

#include <dlfcn.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <random>

namespace llis {
namespace client {

Client::Client(std::string server_name) :
        server_name_(std::move(server_name)),
        c2s_channel_(ipc::c2s_channel_name(server_name_)) {
    generate_client_id();
    create_s2c_channel();
    register_client();
    reconnect_s2c_channel(); // reconnect after getting the permanent client id
    connect_s2c_socket();
}

Client::~Client() {
    close(s2c_socket_);
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
    s2c_socket_ = socket(AF_UNIX, SOCK_DGRAM, 0);

    sockaddr_un addr;
    bzero(&addr, sizeof(addr));
    addr.sun_family = AF_UNIX;
    // TODO: check length. It should not be >= 108
    strcpy(addr.sun_path, ipc::s2c_socket_path(server_name_, client_id_).c_str());

    // TODO: check if it succeeds
    (void)connect(s2c_socket_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
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
    void* handle = dlopen(path.c_str(), RTLD_LAZY);
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
    if (unused_job_instance_refs_.empty()) {
        JobInstanceRefId id = job_instance_refs_.size();
        job_instance_ref.set_id(id);
        job_instance_refs_.push_back(std::move(job_instance_ref));
        return &job_instance_refs_.back();
    } else {
        JobInstanceRefId id = unused_job_instance_refs_.back();
        unused_job_instance_refs_.pop_back();
        job_instance_ref.set_id(id);
        job_instance_refs_[id] = std::move(job_instance_ref);
        return &job_instance_refs_[id];
    }
}

}
}

