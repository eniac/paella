#include <llis/ipc/unix_datagram_socket.h>
#include <llis/ipc/shm_channel.h>
#include <llis/server/scheduler.h>
#include <llis/server/server.h>
#include <llis/ipc/defs.h>
#include <llis/ipc/name_format.h>

#include <chrono>
#include <memory>
#include <thread>
#include <iostream>

namespace llis {
namespace server {

Server::Server(std::string server_name, Scheduler* scheduler) : server_name_(server_name), scheduler_(scheduler), c2s_channel_(ipc::c2s_channel_name(server_name_), CLT2SCHED_CHAN_SIZE), profiler_(&c2s_channel_) {
    LLIS_INFO("Setting up LLIS server...");
    scheduler_->set_server(this);
}

void Server::serve() {
    LLIS_INFO("Starting LLIS server...");
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

        case MsgType::PROFILER_CMD:
            profiler_.handle_cmd();
            break;

        case MsgType::EXIT_CMD:
            exit(0);
            break;
    }
}

void Server::handle_register_client() {
    ClientId client_id;
    c2s_channel_.read(&client_id);
    ipc::ShmChannelCpuWriter s2c_channel_tmp(ipc::s2c_channel_name(server_name_, client_id));

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

    ipc::ShmChannelCpuWriter s2c_channel(ipc::s2c_channel_name(server_name_, client_id), s2c_channel_size);
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
#ifdef PRINT_LAUNCH_JOB_IPC_LATENCY
    static unsigned ipc_latency_skip_num = 0;
    static unsigned long long sum_ipc_latency = 0;
    static unsigned long long num_ipc_latency = 0;

    unsigned long long timestamp;
    c2s_channel_.read(&timestamp);

    if (ipc_latency_skip_num >= 10000) {
        unsigned long long cur_time = std::chrono::steady_clock::now().time_since_epoch().count();

        sum_ipc_latency += cur_time - timestamp;
        ++num_ipc_latency;

        if (num_ipc_latency % 1000 == 0) {
            printf("Avg launch job IPC latency: %f us\n", (double)sum_ipc_latency / (double)num_ipc_latency / (double)1000);
        }
    } else {
        ++ipc_latency_skip_num;
    }
#endif

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
    s2c_socket->write(&msg, sizeof(msg));
}

void Server::notify_job_ends(job::Job* job) {
    ipc::ShmChannelCpuWriter* s2c_channel = client_connections_[job->get_client_id()].get_s2c_channel();
    s2c_channel->write(job->get_remote_ptr());
}

void Server::release_job_instance(std::unique_ptr<job::Job> job) {
    registered_jobs_[job->get_registered_job_id()].release_instance(std::move(job));
}

void Server::update_job_stage_length(job::Job* job, unsigned stage_id, double len) {
    registered_jobs_[job->get_registered_job_id()].update_stage_length(stage_id, len);
}

void Server::set_job_stage_resource(job::Job* job, unsigned stage_id, float res) {
    registered_jobs_[job->get_registered_job_id()].set_stage_resource(stage_id, res);
}

bool Server::has_job_stage_resource(job::Job* job, unsigned stage_id) {
    return registered_jobs_[job->get_registered_job_id()].has_stage_resource(stage_id);
}

double Server::get_job_remaining_length(job::Job* job, unsigned from_stage) const {
    return registered_jobs_[job->get_registered_job_id()].get_remaining_length(from_stage);
}

double Server::get_job_remaining_rl(job::Job* job, unsigned from_stage) const {
    return registered_jobs_[job->get_registered_job_id()].get_remaining_rl(from_stage);
}

const std::vector<double>& Server::get_job_stage_lengths(job::Job* job) const {
    return registered_jobs_[job->get_registered_job_id()].get_stage_lengths();
}

const std::vector<float>& Server::get_job_stage_resources(job::Job* job) const {
    return registered_jobs_[job->get_registered_job_id()].get_stage_resources();
}

}
}

int main(int argc, char** argv) {
    if (argc < 3) {
        LLIS_ERROR("usage: ./server [server name] [unfairness threshold] [ETA] [sched_sleep]");
        exit(1);
    }

    std::string server_name = argv[1];
    float unfairness_threshold = atof(argv[2]);
    float eta = 1;
    if (argc >= 4) {
        eta = atof(argv[3]);
    }
    unsigned sched_sleep = 0;
    if (argc >= 5) {
        sched_sleep = atoi(argv[4]);
    }

    LLIS_INFO("Registering shared memory channel between server and scheduler");
    llis::ipc::ShmChannelCpuWriter ser2sched_channel(SER2SCHED_CHAN_SIZE);

    llis::server::Scheduler scheduler(unfairness_threshold, eta, sched_sleep);
    llis::server::Server server(server_name, &scheduler);

    server.serve();
}
