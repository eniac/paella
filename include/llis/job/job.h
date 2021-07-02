#pragma once

#include <llis/utils/align.h>
#include <llis/ipc/defs.h>
#include <llis/job/finished_block_notifier.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>
#include <chrono>

namespace llis {
namespace job {

class Job {
  public:
    virtual size_t get_input_size() = 0;
    virtual size_t get_output_size() = 0;
    virtual size_t get_param_size() = 0;
    virtual void full_init(void* io_ptr) = 0;
    virtual void init(void* io_ptr) = 0;
    virtual void run_next() = 0;
    virtual bool has_next() const = 0;

    void reset_internal() {
        unset_running();
        unset_started();
        unset_is_mem();
        unset_pre_notify();
        deficit_counter_ = 0;
        cur_stage_ = -1;
        unset_unfit();
    }

    void set_id(JobId id) {
        id_ = id;
    }

    JobId get_id() const {
        return id_;
    }

    int get_cur_stage() const {
        return cur_stage_;
    }

    size_t get_output_offset() {
        return utils::next_aligned_pos(get_input_size(), 8);
    }

    size_t get_pinned_mem_size() {
        return get_output_offset() + utils::next_aligned_pos(get_output_size(), 8);
    }

    unsigned get_num_blocks() const {
        return num_blocks_;
    }

    unsigned get_num_threads_per_block() const {
        return num_threads_per_block_;
    }

    unsigned get_smem_size_per_block() const {
        return smem_size_per_block_;
    }

    unsigned get_num_registers_per_thread() const {
        return num_registers_per_thread_;
    }

    unsigned get_cur_num_blocks() const {
        return cur_num_blocks_;
    }

    unsigned get_cur_num_threads_per_block() const {
        return cur_num_threads_per_block_;
    }

    unsigned get_cur_smem_size_per_block() const {
        return cur_smem_size_per_block_;
    }

    unsigned get_cur_num_registers_per_thread() const {
        return cur_num_registers_per_thread_;
    }

    bool is_mem() const {
        return is_mem_;
    }

    void set_is_mem() {
        is_mem_ = true;
    }

    void unset_is_mem() {
        is_mem_ = false;
    }

    bool is_pre_notify() const {
        return is_pre_notify_;
    }

    void set_pre_notify() {
        is_pre_notify_ = true;
    }

    void unset_pre_notify() {
        is_pre_notify_ = false;
    }

    void set_num_blocks(unsigned num_blocks) {
        num_blocks_ = num_blocks;
    }

    void set_num_threads_per_block(unsigned num_threads_per_block) {
        num_threads_per_block_ = num_threads_per_block;
    }

    void set_smem_size_per_block(unsigned smem_size_per_block) {
        smem_size_per_block_ = smem_size_per_block;
    }

    void set_num_registers_per_thread(unsigned num_registers_per_thread) {
        num_registers_per_thread_ = num_registers_per_thread;
    }

    bool is_running() const {
        return is_running_;
    }

    void set_running(cudaStream_t cuda_stream) {
        is_running_ = true;
        cuda_stream_ = cuda_stream;
        num_running_blocks_ = num_blocks_;
        num_pending_blocks_ = num_blocks_;
        clear_predicted_smids();

        cur_num_blocks_ = num_blocks_;
        cur_num_threads_per_block_ = num_threads_per_block_;
        cur_smem_size_per_block_ = smem_size_per_block_;
        cur_num_registers_per_thread_ = num_registers_per_thread_;

        stage_start_time_ = std::chrono::steady_clock::now();

        ++cur_stage_;
    }

    void set_finished_block_notifier(FinishedBlockNotifier* finished_block_notifier) {
        finished_block_notifier_ = finished_block_notifier;
    }

    FinishedBlockNotifier* get_finished_block_notifier() {
        return finished_block_notifier_;
    }

    std::chrono::time_point<std::chrono::steady_clock> get_stage_start_time() const {
        return stage_start_time_;
    }

    void unset_running() {
        is_running_ = false;
    }

    bool mark_block_start() {
        return --num_pending_blocks_;
    }

    bool mark_block_start(unsigned num) {
        return num_pending_blocks_ -= num;
    }

    void mark_block_finish() {
        num_running_blocks_--;
        if (num_running_blocks_ == 0) {
            unset_running();
        }
    }

    void mark_block_finish(unsigned num) {
        num_running_blocks_ -= num;
        if (num_running_blocks_ <= 0) {
            unset_running();
        }
    }

    cudaStream_t get_cuda_stream() const {
        return cuda_stream_;
    }

    void set_started() {
        has_started_ = true;
    }

    void unset_started() {
        has_started_ = false;
    }

    bool has_started() const {
        return has_started_;
    }

    void set_client_details(ClientId client_id, JobRefId registered_job_id) {
        client_id_ = client_id;
        registered_job_id_ = registered_job_id;
    }

    ClientId get_client_id() const {
        return client_id_;
    }

    JobRefId get_registered_job_id() const {
        return registered_job_id_;
    }

    void* get_remote_ptr() const {
        return remote_ptr_;
    }

    void set_remote_ptr(void* remote_ptr) {
        remote_ptr_ = remote_ptr;
    }

    void add_predicted_smid(unsigned smid) {
        ++predicted_smid_nums_[smid];
    }

    const unsigned* get_predicted_smid_nums() const {
        return predicted_smid_nums_;
    }

    bool has_predicted_smid(unsigned smid) const {
        return predicted_smid_nums_[smid] > 0;
    }

    void dec_predicted_smid(unsigned smid) {
        --predicted_smid_nums_[smid];
    }

    void clear_predicted_smids() {
        std::fill(predicted_smid_nums_, predicted_smid_nums_ + 40, 0);
    }

    void inc_deficit_counter(float val) {
        deficit_counter_ += val;
    }

    float get_deficit_counter() const {
        return deficit_counter_;
    }

    void set_priority(double priority) {
        priority_ = priority;
    }

    double get_priority() const {
        return priority_;
    }

    bool is_unfit() const {
        return is_unfit_;
    }

    void set_unfit() {
        is_unfit_ = true;
    }

    void unset_unfit() {
        is_unfit_ = false;
    }

    void set_stage_lengths_resources(double total, const std::vector<double>& stage_lengths, const std::vector<float>& stage_resources) {
        cur_rl_ = total;
        stage_lengths_ = stage_lengths;
        stage_resources_ = stage_resources;
    }

    double get_cur_rl() const {
        return cur_rl_;
    }

    void dec_cur_rl() {
        if (stage_lengths_.size() > 0) {
            cur_rl_ -= stage_lengths_[cur_stage_ + 1] * stage_resources_[cur_stage_ + 1];
        }
    }

  private:
    bool is_running_ = false;
    cudaStream_t cuda_stream_;
    FinishedBlockNotifier* finished_block_notifier_;

    unsigned num_blocks_;
    unsigned num_threads_per_block_;
    unsigned smem_size_per_block_;
    unsigned num_registers_per_thread_;
    bool is_mem_ = false;
    bool is_pre_notify_ = false;

    unsigned cur_num_blocks_;
    unsigned cur_num_threads_per_block_;
    unsigned cur_smem_size_per_block_;
    unsigned cur_num_registers_per_thread_;

    unsigned num_running_blocks_;
    unsigned num_pending_blocks_;

    unsigned predicted_smid_nums_[40];

    float deficit_counter_ = 0;
    double priority_;

    bool has_started_ = false;

    int cur_stage_ = -1;

    bool is_unfit_ = false;

    std::vector<double> stage_lengths_;
    std::vector<float> stage_resources_;
    double cur_rl_ = 0;

    std::chrono::time_point<std::chrono::steady_clock> stage_start_time_;

    ClientId client_id_;
    JobRefId registered_job_id_;
    void* remote_ptr_;
    JobId id_;
};

}
}

