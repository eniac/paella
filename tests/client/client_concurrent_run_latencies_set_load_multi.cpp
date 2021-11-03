#include "llis/client/job_ref.h"
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

std::vector<std::vector<llis::client::JobInstanceRef*>> job_instance_refss;
std::vector<std::vector<llis::client::JobInstanceRef*>> unused_job_instance_refss;
std::mutex mtx;

int max_num_jobs;

unsigned run_time;
unsigned start_record_time;

unsigned run_time_us;
unsigned start_record_time_us;

void monitor(llis::client::Client* client, const std::string& profile_path) {
    auto very_start_time = std::chrono::steady_clock::now();

    bool has_set_record_exec_time = false;

    while (true) {
        llis::client::JobInstanceRef* job_instance_ref = client->wait();

        auto cur_time = std::chrono::steady_clock::now();

        //client->release_job_instance_ref(job_instance_ref);

        std::unique_lock<std::mutex> lk(mtx);
        unused_job_instance_refss[job_instance_ref->get_job_ref_id()].push_back(job_instance_ref);
        lk.unlock();

        --num_outstanding_jobs;

        double latency = std::chrono::duration<double, std::micro>(cur_time.time_since_epoch()).count() - job_instance_ref->get_start_time();
        double time_elasped = std::chrono::duration<double, std::micro>(cur_time - very_start_time).count();

        if (time_elasped > start_record_time_us) {
            latencies.push_back(latency);
            if (!has_set_record_exec_time) {
                //client->get_profiler_client()->set_record_kernel_info();
                //client->get_profiler_client()->set_record_block_exec_time();
                //client->get_profiler_client()->set_record_kernel_block_mis_alloc();
                //client->get_profiler_client()->set_record_run_next_times();
                has_set_record_exec_time = true;
            }
        }

        if (time_elasped > run_time_us) {
            //client->get_profiler_client()->unset_record_kernel_info();
            //client->get_profiler_client()->unset_record_block_exec_time();
            //client->get_profiler_client()->unset_record_kernel_block_mis_alloc();
            //client->get_profiler_client()->unset_record_run_next_times();
            //client->get_profiler_client()->save(profile_path);
            return;
        }
    }
}

void submit(std::vector<llis::client::JobRef>* job_refs, const std::vector<float>& job_props, double mean_inter_time) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<> d_inter(1. / mean_inter_time);
    std::uniform_real_distribution<float> d_type(0, 1);

    double next_submit_time = 0;

    auto start_time = std::chrono::steady_clock::now();
    double start_time_us = std::chrono::duration<double, std::micro>(start_time.time_since_epoch()).count();

    while (true) {
        auto cur_time = std::chrono::steady_clock::now();
        auto time_elasped = std::chrono::duration<double, std::micro>(cur_time - start_time).count();

        if (time_elasped > run_time_us) {
            return;
        }

        if (time_elasped >= next_submit_time) {
            auto start_time = std::chrono::steady_clock::now();

            while (num_outstanding_jobs >= max_num_jobs);

            unsigned job_type;
            std::unique_lock<std::mutex> lk(mtx);
            do {
                job_type = std::lower_bound(job_props.begin(), job_props.end(), d_type(gen)) - job_props.begin();
            } while (unused_job_instance_refss[job_type].empty());
            llis::client::JobInstanceRef* job_instance_ref = unused_job_instance_refss[job_type].back();
            unused_job_instance_refss[job_type].pop_back();
            lk.unlock();
            job_instance_ref->set_start_time(start_time_us + next_submit_time);
            job_instance_ref->launch();

            next_submit_time += d_inter(gen);

            ++num_outstanding_jobs;
        }
    }
}

int main(int argc, char** argv) {
    const char* server_name = argv[1];
    double mean_inter_time = atof(argv[2]);
    max_num_jobs = atoi(argv[3]);
    run_time = atoi(argv[4]);
    run_time_us = run_time * 1000000;
    start_record_time = atoi(argv[5]);
    start_record_time_us = start_record_time * 1000000;
    const char* output_path = argv[6];
    const char* raw_output_path = argv[7];
    const char* profile_path = argv[8];

    std::vector<const char*> job_paths;
    std::vector<float> job_props;
    std::vector<unsigned> job_max_nums;
    for (unsigned i = 9; i < argc; i += 3) {
        job_paths.push_back(argv[i]);
        if (i == 9) {
            job_props.push_back(std::stof(argv[i + 1]));
        } else {
            job_props.push_back(std::stof(argv[i + 1]) + job_props.back());
        }
        job_max_nums.push_back(std::atoi(argv[i + 2]));
    }

    printf("Before constructing client\n");
    llis::client::Client client(server_name);
    printf("Finished constructing client\n");
    std::vector<llis::client::JobRef> job_refs;
    job_refs.reserve(job_paths.size());
    for (const char* job_path : job_paths) {
        job_refs.push_back(client.register_job(job_path));
    }
    printf("Finished registering\n");

    job_instance_refss.resize(job_refs.size());
    unused_job_instance_refss.resize(job_refs.size());

    auto start_init = std::chrono::steady_clock::now();
    for (unsigned job_type = 0; job_type < job_refs.size(); ++job_type) {
        auto& job_ref = job_refs[job_type];
        auto& job_instance_refs = job_instance_refss[job_type];
        auto& unused_job_instance_refs = unused_job_instance_refss[job_type];

        for (int i = 0; i < job_max_nums[job_type]; ++i) {
            llis::client::JobInstanceRef* job_instance_ref = job_ref.create_instance();
            job_instance_refs.push_back(job_instance_ref);
            unused_job_instance_refs.push_back(job_instance_ref);
            job_instance_ref->launch();
        }
    }
    printf("Finished launching initial jobs\n");
    for (unsigned job_type = 0; job_type < job_refs.size(); ++job_type) {
        for (int i = 0; i < job_max_nums[job_type]; ++i) {
            llis::client::JobInstanceRef* job_instance_ref = client.wait();
            //client.release_job_instance_ref(job_instance_ref);
        }
    }
    auto finish_init = std::chrono::steady_clock::now();
    printf("Finished init. Time taken: %f\n", std::chrono::duration<double, std::micro>(finish_init - start_init).count());

    std::thread monitor_thr(monitor, &client, profile_path);
    std::thread submit_thr(submit, &job_refs, job_props, mean_inter_time);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    pthread_setaffinity_np(monitor_thr.native_handle(), sizeof(cpu_set_t), &cpuset);

    CPU_SET(3, &cpuset);
    pthread_setaffinity_np(submit_thr.native_handle(), sizeof(cpu_set_t), &cpuset);

    std::this_thread::sleep_for(40s);
    //submit_thr.join();

    double throughput = (double)latencies.size() / (double)(run_time - start_record_time);

    std::sort(latencies.begin(), latencies.end());
    
    double mean = 0;

    for (double latency : latencies) {
        mean += (latency / (double)latencies.size());
    }

    double p50 = latencies[latencies.size() / 2];
    double p90 = latencies[latencies.size() * 0.90];
    double p95 = latencies[latencies.size() * 0.95];
    double p99 = latencies[latencies.size() * 0.99];
    double max = *std::max_element(latencies.begin(), latencies.end());

    FILE* fp = fopen(output_path, "a");
    fprintf(fp, "%d,%f,%f,%f,%f,%f,%f,%f\n", (int)mean_inter_time, throughput, mean, p50, p90, p95, p99, max);
    fclose(fp);

    fp = fopen(raw_output_path, "w");
    for (double latency : latencies) {
        fprintf(fp, "%f\n", latency);
    }
    fclose(fp);

    client.kill_server();
}

