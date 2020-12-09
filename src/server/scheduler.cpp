#include <llis/server/scheduler.h>

#include <llis/ipc/shm_primitive_channel.h>
#include <llis/server/server.h>
#include <llis/job/job.h>
#include <llis/job/context.h>
#include <llis/job/instrument_info.h>

#include <sys/socket.h>
#include <sys/un.h>

#include <memory>
#include <string>
#include <algorithm>

namespace llis {
namespace server {

Scheduler::Scheduler() : server_(nullptr), gpu2sched_channel_(10240), cuda_streams_(100) { // TODO: size of the channel must be larger than number of total blocks * 2
    job::Context::set_gpu2sched_channel(&gpu2sched_channel_);

    for (auto& stream : cuda_streams_) {
        cudaStreamCreate(&stream);
    }

    // TODO: query the numbers
    // Some of them can be zero because some smid may not reflect an actual SM
    sm_avails_.resize(40);
    for (auto& sm_avail : sm_avails_) {
        sm_avail.nregs = 65536;
        sm_avail.nthrs = 2048;
        sm_avail.smem = 65536;
        sm_avail.nblocks = 32;
    }

    gpc_num_blocks_.resize(5);
    gpc_next_sms_.resize(5);
}

void Scheduler::set_server(Server* server) {
    server_ = server;
}

void Scheduler::try_handle_block_start_finish() {
    if (gpu2sched_channel_.can_read<job::InstrumentInfo>()) {
        handle_block_start_finish();
    }
}

void Scheduler::handle_block_start_finish() {
    job::InstrumentInfo info = gpu2sched_channel_.read<job::InstrumentInfo>();
    
    if (info.is_start) {
        handle_block_start(info);
    } else {
        handle_block_finish(info);
    }
}

void Scheduler::handle_block_start(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id];

    if (!job->has_predicted_smid(info.smid)) {
        sm_avails_[info.smid].minus(job, 1);
    } else {
        job->dec_predicted_smid(info.smid);
    }

    if (!job->mark_block_start()) {
        const unsigned* predicted_smid_nums = job->get_predicted_smid_nums();
        for (unsigned smid = 0; smid < 40; ++smid) {
            unsigned num = predicted_smid_nums[smid];

            // TODO: handle allocation granularity
            sm_avails_[smid].add(job, num);
        }
    }
}

void Scheduler::handle_block_finish(const job::InstrumentInfo& info) {
    job::Job* job = job_id_to_job_map_[info.job_id];

    job->mark_block_finish();
    if (!job->is_running()) {
        if (!job->has_next()) {
            server_->notify_job_ends(job);
            --num_jobs_;
        }
#ifdef PRINT_NUM_RUNNING_KERNELS
        --num_running_kernels_;
        printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif
        cuda_streams_.push_back(job->get_cuda_stream());
    }

    sm_avails_[info.smid].add(job, 1);

    schedule_job();
}

void Scheduler::handle_new_job(std::unique_ptr<job::Job> job) {
    if (unused_job_id_.empty()) {
        job->set_id(job_id_to_job_map_.size());
        job_id_to_job_map_.push_back(job.get());
    } else {
        job->set_id(unused_job_id_.back());
        unused_job_id_.pop_back();
        job_id_to_job_map_[job->get_id()] = job.get();
    }

    jobs_.push_back(std::move(job));

    ++num_jobs_;

    schedule_job();
}

void Scheduler::schedule_job() {
    schedule_job(true);
    schedule_job(false);
}

void Scheduler::schedule_job(bool is_high) {
    while (!jobs_.empty() && !jobs_.front()->has_next() && !jobs_.front()->is_running()) {
        std::unique_ptr<job::Job> job = std::move(jobs_.front());
        jobs_.pop_front();

        unused_job_id_.push_back(job->get_id());

        server_->release_job_instance(std::move(job));
    }

    if (cuda_streams_.empty()) {
        return;
    }

    // TODO: do actual scheduling. Now it is just running whatever runnable, FIFO
    for (const auto& job : jobs_) {
        if (job->has_next() && !job->is_running()) {
            if (is_high && !job->has_started()) {
                // When scheduling high priority jobs, only schedule those that are started
                continue;
            }

            if (job_fits(job.get())) {
                if (!job->has_started()) {
                    server_->notify_job_starts(job.get());
                    job->set_started();
                }
                job->set_running(cuda_streams_.back());
#ifdef PRINT_NUM_RUNNING_KERNELS
                ++num_running_kernels_;
                printf("num_running_kernels_: %u\n", num_running_kernels_);
#endif
                choose_sms(job.get());
                cuda_streams_.pop_back();
                job::Context::set_current_job(job.get());
                job->run_next();
            }
        }
    }
}

bool Scheduler::job_fits(job::Job* job) {
    unsigned num_avail_blocks = 0;

    for (auto& sm_avail : sm_avails_) {
        if (!sm_avail.is_ok()) {
            continue;
        }

        unsigned tmp = sm_avail.nblocks;

        if (job->get_num_threads_per_block() > 0) {
            tmp = std::min(tmp, sm_avail.nregs / (job->get_num_registers_per_thread() * job->get_num_threads_per_block()));
            tmp = std::min(tmp, sm_avail.nthrs / job->get_num_threads_per_block());
        }

        if (job->get_smem_size_per_block() > 0) {
            tmp = std::min(tmp, sm_avail.smem / job->get_smem_size_per_block());
        }

        num_avail_blocks += tmp;

        if (num_avail_blocks >= job->get_num_blocks()) {
            return true;
        }
    }

    return false;
}

void Scheduler::choose_sms(job::Job* job) {
    unsigned num_blocks = job->get_num_blocks();

    for (unsigned blockId = 0; blockId < num_blocks; ++blockId) {
        unsigned gpc = std::min_element(gpc_num_blocks_.begin(), gpc_num_blocks_.end()) - gpc_num_blocks_.begin();
        unsigned smid = gpc_sms_[gpc][gpc_next_sms_[gpc]];

        ++gpc_num_blocks_[gpc];
        gpc_next_sms_[gpc] = (gpc_next_sms_[gpc] + 1) % 8;

        sm_avails_[smid].minus(job, 1);

        // TODO: handle overusing resources

        job->add_predicted_smid(smid);
    }
}

void Scheduler::update_deficit_counters(job::Job* job_scheduled) {
    float val = 1. / num_jobs_;

    for (auto& job : jobs_) {
        if (job.get() == job_scheduled) {
            job->inc_deficit_counter(1. - val);
        } else {
            job->inc_deficit_counter(val);
        }
    }
}

}
}

