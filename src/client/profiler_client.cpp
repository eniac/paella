#include <llis/client/profiler_client.h>
#include <llis/ipc/defs.h>

namespace llis {
namespace client {

void ProfilerClient::set_record_kernel_info() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::PROFILER_CMD);
    c2s_channel_->write(ProfilerMsgType::SET_RECORD_KERNEL_INFO);

    c2s_channel_->release_writer_lock();
}

void ProfilerClient::unset_record_kernel_info() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::PROFILER_CMD);
    c2s_channel_->write(ProfilerMsgType::UNSET_RECORD_KERNEL_INFO);

    c2s_channel_->release_writer_lock();
}

void ProfilerClient::set_record_block_exec_time() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::PROFILER_CMD);
    c2s_channel_->write(ProfilerMsgType::SET_RECORD_BLOCK_EXEC_TIME);

    c2s_channel_->release_writer_lock();
}

void ProfilerClient::unset_record_block_exec_time() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::PROFILER_CMD);
    c2s_channel_->write(ProfilerMsgType::UNSET_RECORD_BLOCK_EXEC_TIME);

    c2s_channel_->release_writer_lock();
}

void ProfilerClient::set_record_kernel_block_mis_alloc() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::PROFILER_CMD);
    c2s_channel_->write(ProfilerMsgType::SET_RECORD_KERNEL_BLOCK_MIS_ALLOC);

    c2s_channel_->release_writer_lock();
}

void ProfilerClient::unset_record_kernel_block_mis_alloc() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::PROFILER_CMD);
    c2s_channel_->write(ProfilerMsgType::UNSET_RECORD_KERNEL_BLOCK_MIS_ALLOC);

    c2s_channel_->release_writer_lock();
}

void ProfilerClient::set_record_run_next_times() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::PROFILER_CMD);
    c2s_channel_->write(ProfilerMsgType::SET_RECORD_RUN_NEXT_TIMES);

    c2s_channel_->release_writer_lock();
}

void ProfilerClient::unset_record_run_next_times() {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::PROFILER_CMD);
    c2s_channel_->write(ProfilerMsgType::UNSET_RECORD_RUN_NEXT_TIMES);

    c2s_channel_->release_writer_lock();
}

void ProfilerClient::save(const std::string& path) {
    c2s_channel_->acquire_writer_lock();

    c2s_channel_->write(MsgType::PROFILER_CMD);
    c2s_channel_->write(ProfilerMsgType::SAVE);
    c2s_channel_->write(path);

    c2s_channel_->release_writer_lock();
}

}
}

