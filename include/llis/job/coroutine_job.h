#pragma once

#include <boost/coroutine2/all.hpp>
#include <llis/job/job.h>

namespace llis {
namespace job {

class CoroutineJob : public Job {
  public:
    void full_init(void* io_ptr) override;

    void run_next() final;
    virtual void body(boost::coroutines2::coroutine<void>::push_type& coroutine_push) = 0;

    bool has_next() const override {
        return (bool)(*coroutine_pull_);
    }

  private:
    std::unique_ptr<boost::coroutines2::coroutine<void>::pull_type> coroutine_pull_;
};

}
}

