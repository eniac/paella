#pragma once

#include <llis/ipc/atomic_wrapper.h>
#include <llis/ipc/defs.h>

namespace llis {
namespace job {

class InstrumentInfo {
  public:
    uint8_t is_start;
    uint8_t smid;
    uint8_t num;
  private:
    uint8_t status_;
  public:
    JobId job_id;

    CUDA_HOSTDEV InstrumentInfo() {
        set_can_read();
    }

    CUDA_HOSTDEV bool can_read() const {
        return status_ == 1;
    }

    CUDA_HOSTDEV bool can_write() const {
        return status_ == 0;
    }

    CUDA_HOSTDEV void set_can_read() {
        status_ = 1;
    }

    CUDA_HOSTDEV void set_can_write() {
        status_ = 0;
    }
};

#ifdef LLIS_MEASURE_BLOCK_TIME

class BlockStartEndTime {
  private:
    uint8_t status_;
    uint8_t dummy_;
  public:
    uint16_t data[3];

    CUDA_HOSTDEV BlockStartEndTime() {
        set_can_read();
    }

    CUDA_HOSTDEV bool can_read() const {
        return status_ == 1;
    }

    CUDA_HOSTDEV bool can_write() const {
        return status_ == 0;
    }

    CUDA_HOSTDEV void set_can_read() {
        status_ = 1;
    }

    CUDA_HOSTDEV void set_can_write() {
        status_ = 0;
    }
};

#endif

}
}

