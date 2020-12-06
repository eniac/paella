#pragma once

#include <boost/coroutine2/all.hpp>
#include <llis/job/job.h>

namespace llis {
namespace job {

class CoroutineJob : public Job {
  public:
    void full_init(void* io_ptr) override {
        // Note that this will run the coroutine immediately. This is necessary because we need the coroutine to setup resource requirements for the first kernel

        coroutine_pull_ = std::make_unique<boost::coroutines2::coroutine<void>::pull_type>([this](boost::coroutines2::coroutine<void>::push_type& coroutine_push) {
            coroutine_push_ = &coroutine_push;
            body();
        });
    }

    void run_next() final {
        (*coroutine_pull_)();
    }

    virtual void body() = 0;

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

