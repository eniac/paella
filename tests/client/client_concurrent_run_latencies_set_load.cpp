#include <atomic>
#include <llis/client/job_instance_ref.h>
#include <llis/client/client.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <mutex>
#include <ratio>
#include <vector>
#include <thread>
#include <algorithm>
#include <random>

using namespace std::chrono_literals;

std::vector<double> latencies;

std::atomic_int num_outstanding_jobs = 0;

std::vector<llis::client::JobInstanceRef*> job_instance_refs;
std::vector<llis::client::JobInstanceRef*> unused_job_instance_refs;
std::mutex mtx;

int max_num_jobs;

void monitor(llis::client::Client* client, const std::string& profile_path) {
    auto very_start_time = std::chrono::steady_clock::now();

    bool has_set_record_exec_time = false;

    while (true) {
        llis::client::JobInstanceRef* job_instance_ref = client->wait();

        auto cur_time = std::chrono::steady_clock::now();

        //client->release_job_instance_ref(job_instance_ref);

        std::unique_lock<std::mutex> lk(mtx);
        unused_job_instance_refs.push_back(job_instance_ref);
        lk.unlock();

        --num_outstanding_jobs;

        double latency = std::chrono::duration<double, std::micro>(cur_time.time_since_epoch()).count() - job_instance_ref->get_start_time();
        double time_elasped = std::chrono::duration<double, std::micro>(cur_time - very_start_time).count();

        if (time_elasped > 10000000) { // 10s
            latencies.push_back(latency);
            if (!has_set_record_exec_time) {
                client->get_profiler_client()->set_record_kernel_exec_time();
                client->get_profiler_client()->set_record_block_exec_time();
                has_set_record_exec_time = true;
            }
        }

        //printf("time_elasped: %f\n", time_elasped);

        if (time_elasped > 30000000) { // 30s
            client->get_profiler_client()->unset_record_kernel_exec_time();
            client->get_profiler_client()->unset_record_block_exec_time();
            client->get_profiler_client()->save(profile_path);
            return;
        }
    }
}

void submit(llis::client::JobRef* job_ref, double mean_inter_time) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<> d(1. / mean_inter_time);

    double next_submit_time = 0;

    auto start_time = std::chrono::steady_clock::now();
    double start_time_us = std::chrono::duration<double, std::micro>(start_time.time_since_epoch()).count();

    while (true) {
        auto cur_time = std::chrono::steady_clock::now();
        auto time_elasped = std::chrono::duration<double, std::micro>(cur_time - start_time).count();

        if (time_elasped > 30000000) { // 30s
            return;
        }

        if (time_elasped >= next_submit_time) {
            auto start_time = std::chrono::steady_clock::now();

            while (num_outstanding_jobs >= max_num_jobs);

            //llis::client::JobInstanceRef* job_instance_ref = job_ref->create_instance();
            std::unique_lock<std::mutex> lk(mtx);
            llis::client::JobInstanceRef* job_instance_ref = unused_job_instance_refs.back();
            unused_job_instance_refs.pop_back();
            lk.unlock();
            job_instance_ref->set_start_time(start_time_us + next_submit_time);
            job_instance_ref->launch();

            //double start_time_us = std::chrono::duration<double, std::micro>(job_instance_ref->get_start_time() - start_time).count();

            //next_submit_time = start_time_us + next_submit_time_incr;
            next_submit_time += d(gen);

            ++num_outstanding_jobs;
        }
    }
}

int main(int argc, char** argv) {
    const char* server_name = argv[1];
    const char* job_path = argv[2];
    double mean_inter_time = atof(argv[3]);
    max_num_jobs = atoi(argv[4]);
    const char* output_path = argv[5];
    const char* raw_output_path = argv[6];
    const char* profile_path = nullptr;
    if (argc >= 8) {
        profile_path = argv[7];
    }

    llis::client::Client client(server_name);
    llis::client::JobRef job_ref = client.register_job(job_path);

    for (int i = 0; i < max_num_jobs; ++i) {
        llis::client::JobInstanceRef* job_instance_ref = job_ref.create_instance();
        job_instance_refs.push_back(job_instance_ref);
        unused_job_instance_refs.push_back(job_instance_ref);
        job_instance_ref->launch();
    }
    for (int i = 0; i < max_num_jobs; ++i) {
        llis::client::JobInstanceRef* job_instance_ref = client.wait();
        //client.release_job_instance_ref(job_instance_ref);
    }
    printf("Finished init\n");

    std::thread monitor_thr(monitor, &client, profile_path);
    std::thread submit_thr(submit, &job_ref, mean_inter_time);

    //std::this_thread::sleep_for(40s);
    submit_thr.join();

    double throughput = latencies.size() / 20.0;

    std::sort(latencies.begin(), latencies.end());
    
    double mean = 0;

    for (double latency : latencies) {
        mean += (latency / (double)latencies.size());
    }

    double p50 = latencies[latencies.size() / 2];
    double p90 = latencies[latencies.size() * 0.90];
    double p95 = latencies[latencies.size() * 0.95];
    double p99 = latencies[latencies.size() * 0.99];

    FILE* fp = fopen(output_path, "a");
    fprintf(fp, "%d,%f,%f,%f,%f,%f,%f\n", (int)mean_inter_time, throughput, mean, p50, p90, p95, p99);
    fclose(fp);

    fp = fopen(raw_output_path, "w");
    for (double latency : latencies) {
        fprintf(fp, "%f\n", latency);
    }
    fclose(fp);
}

