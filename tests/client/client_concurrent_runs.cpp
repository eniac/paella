#include "llis/client/job_instance_ref.h"
#include <llis/client/client.h>

int main(int argc, char** argv) {
    const char* server_name = argv[1];
    const char* job_path = argv[2];
    int num = atoi(argv[3]);

    llis::client::Client client(server_name);
    llis::client::JobRef job_ref = client.register_job(job_path);

    std::vector<llis::client::JobInstanceRef*> job_instance_refs;
    job_instance_refs.reserve(num);

    for (int i = 0; i < num; ++i) {
        job_instance_refs.push_back(job_ref.create_instance());
        job_instance_refs.back()->launch();
    }
}

