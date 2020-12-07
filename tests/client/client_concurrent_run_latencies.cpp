#include <llis/client/job_instance_ref.h>
#include <llis/client/client.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>

std::vector<double> latencies;

void func(const char* server_name, const char* job_path, int num_streams) {
    llis::client::Client client(server_name);
    llis::client::JobRef job_ref = client.register_job(job_path);

    std::vector<std::chrono::time_point<std::chrono::steady_clock>> start_times;
    start_times.reserve(num_streams);

    auto very_start_time = std::chrono::steady_clock::now();

    for (int sid = 0; sid < num_streams; ++sid) {
        auto start_time = std::chrono::steady_clock::now();

        llis::client::JobInstanceRef* job_instance_ref = job_ref.create_instance();
        //printf("Launching %u\n", job_instance_ref->get_id());
        job_instance_ref->launch();

        start_times.push_back(start_time);
    }

    while (true) {
        llis::client::JobInstanceRef* job_instance_ref = client.wait();
        int sid = job_instance_ref->get_id();

        //printf("sid: %d\n", sid);

        auto cur_time = std::chrono::steady_clock::now();

        auto latency = std::chrono::duration<double, std::micro>(cur_time - start_times[sid]).count();
        auto time_elasped = std::chrono::duration<double, std::micro>(cur_time - very_start_time).count();

        if (time_elasped > 10000000) { // 10s
            latencies.push_back(latency);
        }

        if (time_elasped > 30000000) { // 30s
            return;
        }

        start_times[sid] = cur_time;

        job_instance_ref->launch();
    }
}

int main(int argc, char** argv) {
    const char* server_name = argv[1];
    const char* job_path = argv[2];
    int num_streams = atoi(argv[3]);
    const char* output_path = argv[4];

    func(server_name, job_path, num_streams);

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

