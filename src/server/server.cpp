#include <llis/ipc/unix_datagram_socket.h>
#include <llis/ipc/shm_channel.h>
#include <llis/server/scheduler.h>
#include <llis/server/server.h>
#include <llis/ipc/defs.h>
#include <llis/ipc/name_format.h>

#include <memory>
#include <thread>

namespace llis {
namespace server {

Server::Server(std::string server_name, Scheduer* scheduler) : server_name_(server_name), scheduler_(scheduler), c2s_channel_(ipc::c2s_channel_name(server_name_), 1024) {
    scheduler_->set_server(this);
}

void Server::serve() {
    while (true) {
        try_handle_c2s();
        scheduler_->try_handle_block_start_finish();
    }
}

void Server::try_handle_c2s() {
    if (c2s_channel_.can_read()) {
        handle_c2s();
    }
}

void Server::handle_c2s() {
    MsgType msg_type;
    c2s_channel_.read(&msg_type);

    switch (msg_type) {
        case MsgType::REGISTER_CLIENT:
            handle_register_client();
            break;

        case MsgType::REGISTER_JOB:
            handle_register_job();
            break;

        case MsgType::LAUNCH_JOB:
            handle_launch_job();
            break;

        case MsgType::GROW_POOL:
            handle_grow_pool();
            break;
    }
}

void Server::handle_register_client() {
    ClientId client_id;
    c2s_channel_.read(&client_id);
    ipc::ShmChannel s2c_channel_tmp(ipc::s2c_channel_name(server_name_, client_id));

    ClientConnection* client_connection;
    if (unused_client_connections_.empty()) {
        client_id = client_connections_.size();
        client_connections_.emplace_back(client_id);
        client_connection = &client_connections_.back();
    } else {
        client_id = unused_client_connections_.back();
        unused_client_connections_.pop_back();
        client_connection = &client_connections_[client_id];
    }

    ipc::ShmChannel s2c_channel(ipc::s2c_channel_name(server_name_, client_id), s2c_channel_size);
    client_connection->use_s2c_channel(std::move(s2c_channel));
    
    client_connection->use_s2c_socket(s2c_socket_.connect(ipc::s2c_socket_name(server_name_, client_id)));

    s2c_channel_tmp.write(client_id);
}

void Server::handle_register_job() {
    ClientId client_id;
    c2s_channel_.read(&client_id);

    if (unused_registered_jobs_.empty()) {
        JobRefId registered_job_id = registered_jobs_.size();
        registered_jobs_.emplace_back(registered_job_id, &c2s_channel_, &client_connections_[client_id]);
    } else {
        JobRefId registered_job_id = unused_registered_jobs_.back();
        unused_registered_jobs_.pop_back();
        registered_jobs_[registered_job_id].init(&c2s_channel_, &client_connections_[client_id]);
    }
}

void Server::handle_launch_job() {
    JobRefId registered_job_id;
    c2s_channel_.read(&registered_job_id);

    std::unique_ptr<job::Job> job = registered_jobs_[registered_job_id].create_instance();

    scheduler_->handle_new_job(std::move(job));
}

void Server::handle_grow_pool() {
    JobRefId registered_job_id;
    c2s_channel_.read(&registered_job_id);

    registered_jobs_[registered_job_id].grow_pool();
}

void Server::notify_job_starts(job::Job* job) {
    ipc::UnixDatagramSocket* s2c_socket = client_connections_[job->get_client_id()].get_s2c_socket();
    bool msg = true;
    ssize_t size_written = s2c_socket->write(&msg, sizeof(msg));
}

void Server::notify_job_ends(job::Job* job) {
    ipc::ShmChannel* s2c_channel = client_connections_[job->get_client_id()].get_s2c_channel();
    s2c_channel->write(job->get_job_instance_ref_id());
}

}
}

int main(int argc, char** argv) {
    std::string server_name = argv[1];

    llis::ipc::ShmChannel ser2sched_channel(1024);

    llis::server::Scheduer scheduler;
    llis::server::Server server(server_name, &scheduler);

    server.serve();
}
