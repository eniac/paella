#include <llis/client/client.h>
#include <llis/job.h>

#include <dlfcn.h>
#include <random>

namespace llis {

Client::Client(std::string server_name) :
        server_name_(std::move(server_name)),
        c2s_channel_("server:" + server_name_) {
    generate_client_id();
    create_s2c_channel();
    register_client();
    reconnect_s2c_channel(); // reconnect after getting the permanent client id
}

void Client::generate_client_id() {
    std::random_device rnd;
    client_id_ = rnd();
}

void Client::create_s2c_channel() {
    s2c_channel_.connect("server:" + server_name_ + ":client:" + std::to_string(client_id_), 8);
}

void Client::reconnect_s2c_channel() {
    s2c_channel_.disconnect();
    s2c_channel_.connect("server:" + server_name_ + ":client:" + std::to_string(client_id_));
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
    typedef Job* (*init_job_t)();
    init_job_t init_job = (init_job_t)(dlsym(handle, "init_job"));
    // TODO: error handling

    Job* job = init_job();
    JobRef job_ref(job, this, path);

    return job_ref;
}

}

