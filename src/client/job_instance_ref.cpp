#include <chrono>
#include <llis/job/job.h>
#include <llis/client/job_ref.h>
#include <llis/client/job_instance_ref.h>
#include <llis/client/client.h>

#include <unistd.h>

#include <cmath>

namespace llis {
namespace client {

JobInstanceRef::JobInstanceRef(JobRef* job_ref, IoShmEntry io_shm_entry) : job_ref_(job_ref), io_shm_entry_(io_shm_entry) {
    c2s_channel_ = job_ref_->get_c2s_channel();
}

JobInstanceRef::~JobInstanceRef() {
    // TODO
    //release();
}

void* JobInstanceRef::get_input_ptr() {
    return io_shm_entry_.ptr;
}

void* JobInstanceRef::get_output_ptr() {
    return reinterpret_cast<char*>(io_shm_entry_.ptr) + job_ref_->get_job()->get_input_size();
}

void JobInstanceRef::launch() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::LAUNCH_JOB);
#ifdef PRINT_LAUNCH_JOB_IPC_LATENCY
    unsigned long long cur_time = std::chrono::steady_clock::now().time_since_epoch().count();
    c2s_channel_->write(cur_time);
#endif
    c2s_channel_->write(job_ref_->get_job_ref_id());
    c2s_channel_->write(io_shm_entry_.id);
    c2s_channel_->write(io_shm_entry_.offset);
    c2s_channel_->write(this);

    c2s_channel_->release_writer_lock();
}

void JobInstanceRef::release() {
    job_ref_->release_io_shm_entry(io_shm_entry_);
}

void JobInstanceRef::set_id(JobInstanceRefId id) {
    id_ = id;
}

JobInstanceRefId JobInstanceRef::get_id() const {
    return id_;
}

void JobInstanceRef::set_start_time(double time_point) {
    start_time_ = time_point;
}

double JobInstanceRef::get_start_time() const {
    return start_time_;
}

}
}

