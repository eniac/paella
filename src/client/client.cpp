#include <llis/client/client.h>
#include <llis/job.h>

#include <dlfcn.h>

namespace llis {

//Client::Client(std::string server_name) : c2s_channel_("server:" + server_name) {}
Client::Client(std::string server_name) {}

JobRef Client::register_job(std::string path) {
    void* handle = dlopen(path.c_str(), RTLD_LAZY);
    typedef Job* (*init_job_t)();
    init_job_t init_job = (init_job_t)(dlsym(handle, "init_job"));
    Job* job = init_job();
}

}

