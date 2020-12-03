#include <llis/client/job_instance_ref.h>
#include <llis/client/client.h>

#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    const char* server_name = argv[1];
    const char* job_path = argv[2];
    int num = atoi(argv[3]);

    llis::client::Client client(server_name);
    llis::client::JobRef job_ref = client.register_job(job_path);
    llis::client::JobInstanceRef* job_instance_ref = job_ref.create_instance();

    for (int i = 0; i < num; ++i) {
        auto start_time = std::chrono::steady_clock::now();
        job_instance_ref->launch();
        client.wait();
        auto end_time = std::chrono::steady_clock::now();

        auto time_taken = end_time - start_time;

        std::cout << std::chrono::duration<double, std::micro>(time_taken).count() << std::endl;
    }
}

