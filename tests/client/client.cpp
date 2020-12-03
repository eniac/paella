#include "llis/client/job_instance_ref.h"
#include <llis/client/client.h>

int main(int argc, char** argv) {
    const char* server_name = argv[1];
    const char* job_path = argv[2];

    llis::client::Client client(server_name);
    llis::client::JobRef job_ref = client.register_job(job_path);
    llis::client::JobInstanceRef* job_instance_ref = job_ref.create_instance();
    job_instance_ref->launch();
}

