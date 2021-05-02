#include <chrono>
#include <llis/server/profiler.h>
#include <llis/ipc/defs.h>

namespace llis {
namespace server {

void Profiler::handle_cmd() {
    ProfilerMsgType msg_type;
    c2s_channel_->read(&msg_type);

    switch (msg_type) {
        case ProfilerMsgType::SET_RECORD_KERNEL_INFO:
           kernel_info_flag_ = true; 
           break;

        case ProfilerMsgType::UNSET_RECORD_KERNEL_INFO:
           kernel_info_flag_ = false; 
           break;

        case ProfilerMsgType::SET_RECORD_BLOCK_EXEC_TIME:
           block_exec_times_flag_ = true;
           break;

        case ProfilerMsgType::UNSET_RECORD_BLOCK_EXEC_TIME:
           block_exec_times_flag_ = false;
           break;

        case ProfilerMsgType::SET_RECORD_KERNEL_BLOCK_MIS_ALLOC:
           kernel_block_mis_alloc_flag_ = true;
           break;

        case ProfilerMsgType::UNSET_RECORD_KERNEL_BLOCK_MIS_ALLOC:
           kernel_block_mis_alloc_flag_ = false;
           break;

        case ProfilerMsgType::SET_RECORD_RUN_NEXT_TIMES:
           run_next_times_flag_ = true;
           break;

        case ProfilerMsgType::UNSET_RECORD_RUN_NEXT_TIMES:
           run_next_times_flag_ = false;
           break;

        case ProfilerMsgType::SAVE:
           handle_cmd_save();
           break;
    }
}

void Profiler::handle_cmd_save() {
    std::string path;
    c2s_channel_->read(&path);

    save(path);
}

void Profiler::save(const std::string& path) {
    FILE* fp = fopen((path + "_kernel_info.txt").c_str(), "w");

    for (auto item : kernel_info_) {
        fprintf(fp, "%lu %lu %u %u %u %u\n", std::get<0>(item).time_since_epoch().count(), std::get<1>(item).time_since_epoch().count(), std::get<2>(item), std::get<3>(item), std::get<4>(item), std::get<5>(item));
    }

    fclose(fp);

    fp = fopen((path + "_block_exec_times.txt").c_str(), "w");

    for (auto item : block_exec_times_) {
        fprintf(fp, "%llu %llu\n", item.first, item.second);
    }

    fclose(fp);

    fp = fopen((path + "_kernel_block_mis_alloc.txt").c_str(), "w");

    for (auto item : kernel_block_mis_alloc_) {
        fprintf(fp, "%u %u %u\n", std::get<0>(item), std::get<1>(item), std::get<2>(item));
    }

    fclose(fp);

    fp = fopen((path + "_run_next_times.txt").c_str(), "w");

    for (auto item : run_next_times_) {
        fprintf(fp, "%f %u\n", item.first, item.second);
    }

    fclose(fp);
}

void Profiler::record_kernel_info(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time, unsigned num_blocks, unsigned num_threads_per_block, unsigned smem_size_per_block, unsigned num_registers_per_thread) {
    if (kernel_info_flag_) {
        kernel_info_.emplace_back(start_time, end_time, num_blocks, num_threads_per_block, smem_size_per_block, num_registers_per_thread);
    }
}

void Profiler::record_block_exec_time(unsigned long long start_time, unsigned long long end_time) {
//void Profiler::record_block_exec_time(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time) {
    if (block_exec_times_flag_) {
        block_exec_times_.emplace_back(start_time, end_time);
    }
}

void Profiler::recrod_kernel_block_mis_alloc(unsigned total, unsigned total_wrong_prediction, unsigned total_wrong_prediction_sm) {
    if (kernel_block_mis_alloc_flag_) {
        kernel_block_mis_alloc_.emplace_back(total, total_wrong_prediction, total_wrong_prediction_sm);
    }
}

void Profiler::record_run_next_time(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time, unsigned num_blocks) {
    if (run_next_times_flag_) {
        run_next_times_.emplace_back(std::chrono::duration<double, std::micro>(end_time - start_time).count(), num_blocks);
    }
}

}
}
