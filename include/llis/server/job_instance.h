#pragma once

namespace llis {
namespace server {

class RegisteredJob;

class JobInstance {
  public:
    JobInstance(RegisteredJob* registered_job, void* ptr) {
        printf("JobInstance()\n");
    }
};

}
}

