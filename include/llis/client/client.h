#pragma once

#include "job_ref.h"

#include <cstdint>
#include <string>

class Client {
  public:
    Client(std::string ip, uint16_t port);

    JobRef register_job(std::string path, void* input, void* output);
};

