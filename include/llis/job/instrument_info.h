#pragma once

#include <llis/ipc/atomic_wrapper.h>
#include <llis/ipc/defs.h>

namespace llis {
namespace job {

class InstrumentInfo {
  public:
    uint8_t is_start;
    uint8_t smid;
  private:
    AtomicWrapper<uint16_t, true> status_;
  public:
    JobId job_id;

    CUDA_HOSTDEV InstrumentInfo() {
        set_can_read();
    }

    CUDA_HOSTDEV bool can_read() const {
        return status_.load() == 1;
    }

    CUDA_HOSTDEV bool can_write() const {
        return status_.load() == 0;
    }

    CUDA_HOSTDEV void set_can_read() {
        status_.store(1);
    }

    CUDA_HOSTDEV void set_can_write() {
        status_.store(0);
    }

    CUDA_HOSTDEV bool acquire() {
#ifdef __CUDA_ARCH__
        return (atomicCAS(reinterpret_cast<uint16_t*>(&status_), 0, 2) == 0);
#else
        return false;
#endif
    }
};

#ifdef LLIS_MEASURE_BLOCK_TIME

class BlockStartEndTime {
  private:
    AtomicWrapper<uint16_t, true> status_;
  public:
    uint16_t data[3];

    CUDA_HOSTDEV BlockStartEndTime() {
        set_can_read();
    }

    CUDA_HOSTDEV bool can_read() const {
        return status_.load() == 1;
    }

    CUDA_HOSTDEV bool can_write() const {
        return status_.load() == 0;
    }

    CUDA_HOSTDEV void set_can_read() {
        status_.store(1);
    }

    CUDA_HOSTDEV void set_can_write() {
        status_.store(0);
    }

    CUDA_HOSTDEV bool acquire() {
#ifdef __CUDA_ARCH__
        return (atomicCAS(reinterpret_cast<uint16_t*>(&status_), 0, 2) == 0);
#else
        return false;
#endif
    }
};

#endif

}
}

