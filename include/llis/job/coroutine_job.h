#pragma once

#include <boost/coroutine2/all.hpp>
#include <llis/job/job.h>

namespace llis {
namespace job {

class CoroutineJob : public Job {
  public:
    void full_init(void* io_ptr) final {
        one_time_init();
        init(io_ptr);
    }

    void init(void* io_ptr) final {
        // Note that this will run the coroutine immediately. This is necessary because we need the coroutine to setup resource requirements for the first kernel

        coroutine_pull_ = std::make_unique<boost::coroutines2::coroutine<void>::pull_type>([this, io_ptr](boost::coroutines2::coroutine<void>::push_type& coroutine_push) {
            coroutine_push_ = &coroutine_push;
            body(io_ptr);
        });
    }

    void run_next() final {
        (*coroutine_pull_)();
    }

    virtual void one_time_init() = 0;

    virtual void body(void* io_ptr) = 0;

    bool has_next() const final {
        return (bool)(*coroutine_pull_);
    }

    void yield() {
        (*coroutine_push_)();
    }

  private:
    std::unique_ptr<boost::coroutines2::coroutine<void>::pull_type> coroutine_pull_;
    boost::coroutines2::coroutine<void>::push_type* coroutine_push_;
};

}
}

