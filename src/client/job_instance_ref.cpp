#include <llis/job/job.h>
#include <llis/client/job_ref.h>
#include <llis/client/job_instance_ref.h>

namespace llis {
namespace client {

JobInstanceRef::JobInstanceRef(JobRef* job_ref, IoShmEntry io_shm_entry) : job_ref_(job_ref), io_shm_entry_(io_shm_entry) {
    s2c_channel_ = job_ref_->get_s2c_channel();
    c2s_channel_ = job_ref_->get_c2s_channel();
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
    c2s_channel_->write(job_ref_->get_job_ref_id());
    c2s_channel_->write(io_shm_entry_.id);
    c2s_channel_->write(io_shm_entry_.offset);

    c2s_channel_->release_writer_lock();
}

void JobInstanceRef::wait() {
    // TODO
}

}
}

