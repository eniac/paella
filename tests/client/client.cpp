#include "llis/client/job_instance_ref.h"
#include <llis/client/client.h>

int main() {
    llis::client::Client client("server");
    llis::client::JobRef job_ref = client.register_job("/data/kelvin/Programming/llis/release/jobs/helloworld/libjob_helloworld.so");
    llis::client::JobInstanceRef job_instance_ref = job_ref.create_instance();
    job_instance_ref.launch();
}

