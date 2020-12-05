#include <boost/coroutine2/coroutine.hpp>
#include <llis/job/coroutine_job.h>

namespace llis {
namespace job {

void CoroutineJob::full_init(void* io_ptr) {
    // Note that this will run the coroutine immediately. This is necessary because we need the coroutine to setup resource requirements for the first kernel

    coroutine_pull_ = std::make_unique<boost::coroutines2::coroutine<void>::pull_type>([this](boost::coroutines2::coroutine<void>::push_type& coroutine_push){
        body(coroutine_push);
    });
}

void CoroutineJob::run_next() {
    (*coroutine_pull_)();

}

}
}

