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

using namespace std::chrono_literals;

std::vector<double> latencies;

std::atomic_int num_outstanding_jobs = 0;

std::vector<llis::client::JobInstanceRef*> job_instance_refs;
std::vector<llis::client::JobInstanceRef*> unused_job_instance_refs;
std::mutex mtx;

void monitor(llis::client::Client* client) {
    auto very_start_time = std::chrono::steady_clock::now();

    while (true) {
        llis::client::JobInstanceRef* job_instance_ref = client->wait();

        auto cur_time = std::chrono::steady_clock::now();

        //client->release_job_instance_ref(job_instance_ref);

        std::unique_lock<std::mutex> lk(mtx);
        unused_job_instance_refs.push_back(job_instance_ref);
        lk.unlock();

        --num_outstanding_jobs;

        double latency = std::chrono::duration<double, std::micro>(cur_time - job_instance_ref->get_start_time()).count();
        double time_elasped = std::chrono::duration<double, std::micro>(cur_time - very_start_time).count();

        if (time_elasped > 10000000) { // 10s
            latencies.push_back(latency);
        }

        if (time_elasped > 30000000) { // 30s
            return;
        }
    }
}

void submit(llis::client::JobRef* job_ref, double next_submit_time_incr) {
    double next_submit_time = 0;

    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        auto cur_time = std::chrono::steady_clock::now();
        auto time_elasped = std::chrono::duration<double, std::micro>(cur_time - start_time).count();

        if (time_elasped > 30000000) { // 30s
            return;
        }

        if (time_elasped >= next_submit_time) {
            while (num_outstanding_jobs >= 10);

            //llis::client::JobInstanceRef* job_instance_ref = job_ref->create_instance();
            std::unique_lock<std::mutex> lk(mtx);
            llis::client::JobInstanceRef* job_instance_ref = unused_job_instance_refs.back();
            unused_job_instance_refs.pop_back();
            lk.unlock();
            job_instance_ref->record_start_time();
            job_instance_ref->launch();

            double start_time_us = std::chrono::duration<double, std::micro>(job_instance_ref->get_start_time() - start_time).count();

            next_submit_time = start_time_us + next_submit_time_incr;

            ++num_outstanding_jobs;
        }
    }
}

int main(int argc, char** argv) {
    const char* server_name = argv[1];
    const char* job_path = argv[2];
    double next_submit_time_incr = atof(argv[3]);
    const char* output_path = argv[4];

    llis::client::Client client(server_name);
    llis::client::JobRef job_ref = client.register_job(job_path);

    for (int i = 0; i < 100; ++i) {
        llis::client::JobInstanceRef* job_instance_ref = job_ref.create_instance();
        job_instance_refs.push_back(job_instance_ref);
        unused_job_instance_refs.push_back(job_instance_ref);
        job_instance_ref->launch();
    }
    for (int i = 0; i < 100; ++i) {
        llis::client::JobInstanceRef* job_instance_ref = client.wait();
        //client.release_job_instance_ref(job_instance_ref);
    }
    //printf("Finished init\n");

    std::thread monitor_thr(monitor, &client);
    std::thread submit_thr(submit, &job_ref, next_submit_time_incr);

    std::this_thread::sleep_for(40s);

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
    fprintf(fp, "%f,%f,%f,%f,%f,%f\n", throughput, mean, p50, p90, p95, p99);
    fclose(fp);
}

