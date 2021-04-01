#include <llis/client/job_instance_ref.h>
#include <llis/client/client.h>

#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    const char* server_name = argv[1];
    const char* job_path = argv[2];
    int num = atoi(argv[3]);
    const char* profile_path = nullptr;
    if (argc >= 5) {
        profile_path = argv[4];
    }

    llis::client::Client client(server_name);
    llis::client::JobRef job_ref = client.register_job(job_path);
    llis::client::JobInstanceRef* job_instance_ref = job_ref.create_instance();

    client.get_profiler_client()->set_record_kernel_info();

    for (int i = 0; i < num; ++i) {
        auto start_time = std::chrono::steady_clock::now();
        job_instance_ref->launch();
        client.wait();
        auto end_time = std::chrono::steady_clock::now();

        auto time_taken = end_time - start_time;

        std::cout << std::chrono::duration<double, std::micro>(time_taken).count() << std::endl;
    }

    client.get_profiler_client()->unset_record_kernel_info();
    if (profile_path != nullptr) {
        client.get_profiler_client()->save(profile_path);
    }
}

