#include <llis/server/profiler.h>
#include <llis/ipc/defs.h>

namespace llis {
namespace server {

void Profiler::handle_cmd() {
    ProfilerMsgType msg_type;
    c2s_channel_->read(&msg_type);

    switch (msg_type) {
        case ProfilerMsgType::SET_RECORD_KERNEL_EXEC_TIME:
           kernel_exec_times_flag_ = true; 
           break;

        case ProfilerMsgType::UNSET_RECORD_KERNEL_EXEC_TIME:
           kernel_exec_times_flag_ = false; 
           break;

        case ProfilerMsgType::SET_RECORD_BLOCK_EXEC_TIME:
           block_exec_times_flag_ = true; 
           break;

        case ProfilerMsgType::UNSET_RECORD_BLOCK_EXEC_TIME:
           block_exec_times_flag_ = false; 
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
    FILE* fp = fopen((path + "_kernel_exec_times.txt").c_str(), "w");

    for (auto item : kernel_exec_times_) {
        fprintf(fp, "%lu %lu\n", item.first.time_since_epoch().count(), item.second.time_since_epoch().count());
    }

    fclose(fp);

    fp = fopen((path + "_block_exec_times.txt").c_str(), "w");

    for (auto item : block_exec_times_) {
        fprintf(fp, "%llu %llu\n", item.first, item.second);
    }

    fclose(fp);
}

void Profiler::record_kernel_exec_time(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time) {
    if (kernel_exec_times_flag_) {
        kernel_exec_times_.emplace_back(start_time, end_time);
    }
}

void Profiler::record_block_exec_time(unsigned long long start_time, unsigned long long end_time) {
//void Profiler::record_block_exec_time(const std::chrono::time_point<std::chrono::steady_clock>& start_time, const std::chrono::time_point<std::chrono::steady_clock>& end_time) {
    if (block_exec_times_flag_) {
        block_exec_times_.emplace_back(start_time, end_time);
    }
}

}
}

