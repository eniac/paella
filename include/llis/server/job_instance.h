#pragma once

#include <llis/server/registered_job.h>

namespace llis {
namespace server {

class JobInstance {
  public:
    JobInstance(RegisteredJob* registered_job, void* ptr) {
        printf("JobInstance()\n");
    }
};

}
}

