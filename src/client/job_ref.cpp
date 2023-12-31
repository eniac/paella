#include <llis/utils/align.h>
#include <llis/client/job_ref.h>
#include <llis/client/client.h>

#include <memory>
#include <string>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace llis {
namespace client {

JobRef::JobRef(std::unique_ptr<job::Job> job, Client* client, std::string model_path) : job_(std::move(job)), client_(client), model_path_(model_path) {
    pinned_mem_size_ = job_->get_pinned_mem_size();
    param_size_ = job_->get_param_size();

    s2c_channel_ = client_->get_s2c_channel();
    c2s_channel_ = client_->get_c2s_channel();
    client_id_ = client_->get_client_id();

    shm_name_ = "llis:shm:" + std::to_string(client_id_) + ":" + std::to_string((intptr_t)this);
    shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0600);

    register_job();

    grow_pool(2); // Initialize the pool to support 2 concurrent requests
}

JobRef::~JobRef() {
    // Temporarily comment the clean-up.
    // TODO: make clean-up work properly
    //if (shm_fd_ != -1) {
    //    close(shm_fd_);
    //    shm_unlink(shm_name_.c_str());
    //}
}

void JobRef::register_job() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::REGISTER_JOB);
    c2s_channel_->write(client_id_);
    c2s_channel_->write(model_path_);
    c2s_channel_->write(shm_name_);

    c2s_channel_->release_writer_lock();

    s2c_channel_->read(&job_ref_id_);
}

JobInstanceRef* JobRef::create_instance() {
    if (pinned_mem_free_list_.empty()) {
        grow_pool();
    }

    const IoShmEntry& io_shm_entry = pinned_mem_free_list_.back();
    JobInstanceRef job_instance_ref(this, io_shm_entry);
    pinned_mem_free_list_.pop_back();

    return client_->add_job_instance_ref(std::move(job_instance_ref));
}

void JobRef::release_io_shm_entry(IoShmEntry io_shm_entry) {
    pinned_mem_free_list_.push_back(io_shm_entry);
}

void JobRef::grow_pool(size_t least_num_new_entries) {
    size_t least_num_new_bytes = least_num_new_entries * pinned_mem_size_;
    // FIXME: it sometimes breaks when we allocate different size of pool for different jobs. Now workaround by allocating a large enough constant size pool
    //size_t num_new_bytes = utils::next_aligned_pos(least_num_new_bytes, sysconf(_SC_PAGE_SIZE));
    size_t num_new_bytes = 30638080;
    size_t num_new_entries = num_new_bytes / pinned_mem_size_;

    size_t old_pool_size_in_bytes = pool_size_in_bytes_;

    pool_size_ += num_new_entries;
    pool_size_in_bytes_ += num_new_bytes;

    ftruncate(shm_fd_, pool_size_in_bytes_);

    void* shm_ptr = mmap(nullptr, num_new_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, old_pool_size_in_bytes);

    pinned_mem_list_.push_back(shm_ptr);

    size_t old_num_free_entries = pinned_mem_free_list_.size();
    pinned_mem_free_list_.resize(old_num_free_entries + num_new_entries);
    for (size_t i = old_num_free_entries; i < pinned_mem_free_list_.size(); ++i) {
        pinned_mem_free_list_[i].id = pinned_mem_list_.size() - 1;
        pinned_mem_free_list_[i].offset = (i - old_num_free_entries) * pinned_mem_size_;
        pinned_mem_free_list_[i].ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(shm_ptr) + pinned_mem_free_list_[i].offset);
    }

    c2s_channel_->acquire_writer_lock();
    c2s_channel_->write(MsgType::GROW_POOL);
    c2s_channel_->write(job_ref_id_);
    c2s_channel_->write(num_new_bytes);
    c2s_channel_->release_writer_lock();
}

void JobRef::grow_pool() {
    grow_pool(pool_size_); // double the size
}

}
}

