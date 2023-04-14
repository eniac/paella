#define DIS_LN
#define SUBMIT_PREGEN

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cuda_runtime.h>

#include <getopt.h>

#include <atomic>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <mutex>
#include <ratio>
#include <vector>
#include <thread>
#include <algorithm>
#include <random>
#include <queue>

using namespace std::chrono_literals;

class MyModule {
  public:
    MyModule(tvm::runtime::Module* mod_factory, const std::vector<int64_t>& input_dim, const std::vector<int64_t>& output_dim, unsigned job_type) {
        DLDevice ctx_gpu{kDLCUDA, 0};
        DLDevice ctx_cpu{kDLCPU, 0};

        gmod_ = mod_factory->GetFunction("default")(ctx_gpu);
        set_input_ = gmod_.GetFunction("set_input");
        get_output_ = gmod_.GetFunction("get_output");
        run_ = gmod_.GetFunction("run");

        input_ = tvm::runtime::NDArray::Empty(input_dim, DLDataType{kDLFloat, 32, 1}, ctx_cpu);
        output_ = tvm::runtime::NDArray::Empty(output_dim, DLDataType{kDLFloat, 32, 1}, ctx_cpu);

        input_dev_ = tvm::runtime::NDArray::Empty(input_dim, DLDataType{kDLFloat, 32, 1}, ctx_gpu);
        output_dev_ = tvm::runtime::NDArray::Empty(output_dim, DLDataType{kDLFloat, 32, 1}, ctx_gpu);

        job_type_ = job_type;
    }

    void run(cudaStream_t stream) {
        stream_ = stream;

        TVMSetStream(kDLCUDA, 0, stream_);

        input_dev_.CopyFrom(input_);

        set_input_(0, input_dev_);
        run_();
        get_output_(0, output_dev_);

        output_.CopyFrom(output_dev_);
    }

    unsigned get_job_type() const {
        return job_type_;
    }

    void set_start_time(double start_time) {
        start_time_ = start_time;
    }

    double get_start_time() const {
        return start_time_;
    }

    cudaStream_t get_stream() const {
        return stream_;
    }

    void unset_stream() {
        stream_ = nullptr;
    }

    bool is_done() {
        return cudaStreamQuery(stream_) == cudaSuccess;
    }

  private:
    tvm::runtime::Module gmod_;

    tvm::runtime::PackedFunc set_input_;
    tvm::runtime::PackedFunc get_output_;
    tvm::runtime::PackedFunc run_;

    tvm::runtime::NDArray input_;
    tvm::runtime::NDArray output_;
    tvm::runtime::NDArray input_dev_;
    tvm::runtime::NDArray output_dev_;

    unsigned job_type_;
    double start_time_;
    cudaStream_t stream_ = nullptr;
};

std::vector<double> latencies;
std::vector<std::vector<double>> latencies_per_type;

std::vector<unsigned> num_outstanding_jobs_timeline;
std::vector<unsigned> num_outstanding_jobs_per_type;

std::vector<MyModule> modules;

std::vector<cudaStream_t> unused_streams;
std::vector<std::vector<MyModule*>> unused_modules;
std::atomic_int num_outstanding_jobs = 0;
std::vector<std::queue<double>> due_jobs;
std::atomic_uint* unused_modules_nums;
std::mutex mtx;

std::chrono::time_point<std::chrono::steady_clock> start_time;
std::chrono::time_point<std::chrono::steady_clock> end_time;

int max_num_outstanding_jobs;
unsigned start_record_num;
double mean_inter_time;
#ifdef DIS_LN
double log_normal_sigma = -1;
#endif
#ifdef SUBMIT_DIS
std::vector<float> job_props_cum;
#endif
unsigned num_jobs;
unsigned seed;
#ifdef SUBMIT_PREGEN
std::queue<std::pair<unsigned, double>> pregen_submissions;
#endif
unsigned long long preset_start_time = 0;

void record_num_oustanding_jobs_timeline(double timestamp) {
    num_outstanding_jobs_timeline.push_back(timestamp);
    for (unsigned i = 0; i < num_outstanding_jobs_per_type.size(); ++i) {
        num_outstanding_jobs_timeline.push_back(num_outstanding_jobs_per_type[i]);
    }
}

void launch_job(unsigned job_type, double expected_start_time) {
    unused_modules_nums[job_type].fetch_sub(1, std::memory_order_release);

    MyModule* mod = unused_modules[job_type].back();
    unused_modules[job_type].pop_back();

    cudaStream_t stream = unused_streams.back();
    unused_streams.pop_back();

    ++num_outstanding_jobs_per_type[job_type];

    mod->set_start_time(expected_start_time);
    mod->run(stream);
}

#ifdef SUBMIT_DIS
std::pair<unsigned, double> next_submission() { // returns (job_type, interval)
    static std::mt19937 gen(seed);
    static std::uniform_real_distribution<float> d_type(0, 1);
#ifdef DIS_EXP
    static std::exponential_distribution<> d_inter(1. / mean_inter_time);
#else // if DIS_LN
    static const double log_normal_mu = log(mean_inter_time) - log_normal_sigma * log_normal_sigma / 2;
    static std::lognormal_distribution<> d_inter(log_normal_mu, log_normal_sigma);
#endif

    unsigned job_type;
    //do {
    //    job_type = std::lower_bound(job_props_cum.begin(), job_props_cum.end(), d_type(gen)) - job_props_cum.begin();
    //} while (unused_modules_nums[job_type].load(std::memory_order_acquire) == 0);
    job_type = std::lower_bound(job_props_cum.begin(), job_props_cum.end(), d_type(gen)) - job_props_cum.begin();

    double next_inter = d_inter(gen);

    return std::make_pair(job_type, next_inter);
}
#endif

#ifdef SUBMIT_PREGEN
std::pair<unsigned, double> next_submission() { // returns (job_type, interval)
    std::pair<unsigned, double> ret = pregen_submissions.front();
    pregen_submissions.pop();
    return ret;
}
#endif

void monitor() {
    auto very_start_time = std::chrono::steady_clock::now();

    bool has_set_record_exec_time = false;

    for (unsigned num_waited_jobs = 0; num_waited_jobs < num_jobs; ++num_waited_jobs) {
        MyModule* mod = nullptr;
        do {
            for (unsigned i = 0; i < modules.size(); ++i) {
                if (modules[i].get_stream() && modules[i].is_done()) {
                    mod = &modules[i];
                    break;
                }
            }
        } while (!mod);

        auto cur_time = std::chrono::steady_clock::now();

        double latency = std::chrono::duration<double, std::micro>(cur_time.time_since_epoch()).count() - mod->get_start_time();
        double time_elasped = std::chrono::duration<double, std::micro>(cur_time - very_start_time).count();

        unsigned job_type = mod->get_job_type();

        std::unique_lock<std::mutex> lk(mtx);

        unused_modules[job_type].push_back(mod);
        unused_modules_nums[job_type].fetch_add(1, std::memory_order_release);

        unused_streams.push_back(mod->get_stream());
        mod->unset_stream();

        --num_outstanding_jobs_per_type[job_type];

        record_num_oustanding_jobs_timeline(time_elasped);

        num_outstanding_jobs.fetch_sub(1, std::memory_order_release);

        while (!due_jobs[job_type].empty() && !unused_modules[job_type].empty() && !unused_streams.empty()) {
            launch_job(job_type, due_jobs[job_type].front());
            due_jobs[job_type].pop();
        }

        lk.unlock();

        if (num_waited_jobs >= start_record_num) {
            latencies.push_back(latency);
            latencies_per_type[job_type].push_back(latency);
            if (!has_set_record_exec_time) {
                has_set_record_exec_time = true;
            }
        }
    }

    end_time = std::chrono::steady_clock::now();
}

void submit() {
    double next_submit_time = 0;

    if (preset_start_time != 0) {
        while (std::chrono::system_clock::now().time_since_epoch().count() < preset_start_time);
    }
    start_time = std::chrono::steady_clock::now();
    double start_time_us = std::chrono::duration<double, std::micro>(start_time.time_since_epoch()).count();

    printf("Start submitting actual jobs\n");

    unsigned num_subitted_jobs = 0;

    unsigned num_queue_full = 0;
    unsigned num_per_job_full = 0;

    while (true) {
        auto cur_time = std::chrono::steady_clock::now();
        auto time_elasped = std::chrono::duration<double, std::micro>(cur_time - start_time).count();

        if (num_subitted_jobs >= num_jobs) {
            printf("Finished submitting all jobs. Time taken: %f us\n", std::chrono::duration<double, std::micro>(cur_time - start_time).count());
            printf("%f num_outstanding_jobs >= max_num_outstanding_jobs times: %u\n", mean_inter_time, num_queue_full);
            printf("%f Per job full times: %u\n", mean_inter_time, num_per_job_full);
            return;
        }

        if (time_elasped >= next_submit_time) {
            if (num_outstanding_jobs.load(std::memory_order_acquire) >= max_num_outstanding_jobs) {
                ++num_queue_full;
                while (num_outstanding_jobs.load(std::memory_order_acquire) >= max_num_outstanding_jobs);
            }

            auto [job_type, next_inter] = next_submission();

            std::unique_lock<std::mutex> lk(mtx);

            if (unused_modules_nums[job_type].load(std::memory_order_acquire) == 0) {
                ++num_per_job_full;
                due_jobs[job_type].push(start_time_us + next_submit_time);
            } else {
                launch_job(job_type, start_time_us + next_submit_time);

                record_num_oustanding_jobs_timeline(time_elasped);
            }

            lk.unlock();

            num_outstanding_jobs.fetch_add(1, std::memory_order_release);
            ++num_subitted_jobs;

            next_submit_time += next_inter;
        }
    }
}

void print_latency_stats(FILE* fp, std::vector<double>* latencies) {
    std::sort(latencies->begin(), latencies->end());

    double mean = 0;
    double mean_sqr = 0;

    for (double latency : *latencies) {
        mean += (latency / (double)latencies->size());
        mean_sqr += latency * latency / (double)latencies->size();
    }

    double sd = sqrt((mean_sqr - mean * mean) * ((double)latencies->size() / ((double)latencies->size() - 1)));

    double p50 = (*latencies)[latencies->size() / 2];
    double p90 = (*latencies)[latencies->size() * 0.90];
    double p95 = (*latencies)[latencies->size() * 0.95];
    double p99 = (*latencies)[latencies->size() * 0.99];
    double max = *std::max_element(latencies->begin(), latencies->end());

    fprintf(fp, ",%f,%f,%f,%f,%f,%f,%f", mean, p50, p90, p95, p99, max, sd);
}

int main(int argc, char** argv) {
    cudaSetDevice(0);

#ifdef SUBMIT_DIS
    mean_inter_time = -1;
#ifdef DIS_LN
    log_normal_sigma = -1;
#endif
    num_jobs = -1;
    seed = -1;
#endif
    max_num_outstanding_jobs = -1;
    start_record_num = 0;
    const char* output_path_prefix;
#ifdef SUBMIT_PREGEN
    const char* pregen_input_prefix;
    const char* pregen_job_id = nullptr;
#endif
    double fairness_threshold = -1;
    unsigned sched_sleep = 0;

    int mean_inter_time_n = 0;
    int log_normal_sigma_n = 0;
    int max_num_outstanding_jobs_n = 0;
    int num_jobs_n = 0;
    int start_record_num_n = 0;
    int seed_n = 0;
    int fairness_n = 0;
    int sched_sleep_n = 0;

    int mean_inter_time_g = 0;
    int log_normal_sigma_g = 0;
    int max_num_outstanding_jobs_g = 0;
    int num_jobs_g = 0;
    int start_record_num_g = 0;
    int seed_g = 0;
    int fairness_g = 0;
    int sched_sleep_g = 0;

    while (true) {
        static struct option long_options[] = {
            {"iat",                required_argument, 0, 'i'},
#ifdef DIS_LN
            {"ln_sigma",           required_argument, 0, 'l'},
#endif
            {"num_jobs",           required_argument, 0, 'n'},
            {"seed",               required_argument, 0, 's'},
            {"concurrency",        required_argument, 0, 'c'},
            {"start_record_num",   required_argument, 0, 'r'},
            {"prefix",             required_argument, 0, 'p'},
#ifdef SUBMIT_PREGEN
            {"pregen_prefix",      required_argument, 0, 'g'},
            {"pregen_job_id",      required_argument, 0, 'j'},
#endif
            {"fairness",           required_argument, 0, 'f'},
            {"sched_sleep",        required_argument, 0, 'h'},
            {"preset_start_time",  required_argument, 0, 't'},
            
            {"iat_n",              no_argument,       &mean_inter_time_n, 1},
#ifdef DIS_LN
            {"ln_sigma_n",         no_argument,       &log_normal_sigma_n, 1},
#endif
            {"num_jobs_n",         no_argument,       &num_jobs_n, 1},
            {"seed_n",             no_argument,       &seed_n, 1},
            {"concurrency_n",      no_argument,       &max_num_outstanding_jobs_n, 1},
            {"start_record_num_n", no_argument,       &start_record_num_n, 1},
            {"fairness_n",         no_argument,       &fairness_n, 1},
            {"sched_sleep_n",      no_argument,       &sched_sleep_n, 1},
            
            {"iat_g",              no_argument,       &mean_inter_time_g, 1},
#ifdef DIS_LN                                                                         
            {"ln_sigma_g",         no_argument,       &log_normal_sigma_g, 1},
#endif                                                                                
            {"concurrency_g",      no_argument,       &max_num_outstanding_jobs_g, 1},
            {"num_jobs_g",         no_argument,       &num_jobs_g, 1},
            {"start_record_num_g", no_argument,       &start_record_num_g, 1},
            {"seed_g",             no_argument,       &seed_g, 1},
            {"fairness_g",         no_argument,       &fairness_g, 1},
            {"sched_sleep_g",      no_argument,       &sched_sleep_g, 1},
 
            {0,                    0,                 0,  0 }
        };

        int opt_idx = 0;
#ifdef SUBMIT_DIS
        int opt_val = getopt_long(argc, argv, "m:i:l:c:n:r:s:p:f:t:", long_options, &opt_idx);
#endif
#ifdef SUBMIT_PREGEN
        int opt_val = getopt_long(argc, argv, "m:i:l:c:n:r:s:p:g:j:f:t:", long_options, &opt_idx);
#endif

        if (opt_val == -1) {
            break;
        }

        switch (opt_val) {
            case 0:
                break;

            case 'i':
                mean_inter_time = atof(optarg);
                break;
#ifdef DIS_LN
            case 'l':
                log_normal_sigma = atof(optarg);
                break;
#endif
            case 'c':
                max_num_outstanding_jobs = atoi(optarg);
                break;
            case 'n':
                num_jobs = atoi(optarg);
                break;
            case 'r':
                start_record_num = atoi(optarg);
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'p':
                output_path_prefix = optarg;
                break;
#ifdef SUBMIT_PREGEN
            case 'g':
                pregen_input_prefix = optarg;
                break;
            case 'j':
                pregen_job_id = optarg;
                break;
#endif
            case 'f':
                fairness_threshold = atof(optarg);
                break;
            case 'h':
                sched_sleep = atoi(optarg);
                break;
            case 't':
                preset_start_time = strtoull(optarg, nullptr, 10);
                break;
        }
    }

#ifdef SUBMIT_PREGEN
    std::stringstream pregen_input_path;
    pregen_input_path << pregen_input_prefix;
    pregen_input_path << "_iat" << mean_inter_time;
    pregen_input_path << "_lns" << log_normal_sigma;
    pregen_input_path << "_n" << num_jobs;
    pregen_input_path << "_seed" << seed;
    if (pregen_job_id != nullptr) {
        pregen_input_path << "_job" << pregen_job_id;
    }
    pregen_input_path << ".txt";
    printf("pregen_input_path: %s\n", pregen_input_path.str().c_str());

    FILE* pregen_fp = fopen(pregen_input_path.str().c_str(), "r");
    while (!feof(pregen_fp)) {
        unsigned job_type;
        fscanf(pregen_fp, "%u", &job_type);
        double inter;
        fscanf(pregen_fp, "%f", &inter);
        pregen_submissions.emplace(job_type, inter);
    }
    
    num_jobs = pregen_submissions.size();
#endif

    std::stringstream output_path_stats_prefix;
    output_path_stats_prefix << output_path_prefix;

    // Create a output path prefix with all requested parameters except the one that is grouped
    if (mean_inter_time_n && !mean_inter_time_g) {
        output_path_stats_prefix << "_iat" << mean_inter_time;
    }
    if (log_normal_sigma_n && !log_normal_sigma_g) {
        output_path_stats_prefix << "_lns" << log_normal_sigma;
    }
    if (max_num_outstanding_jobs_n && !max_num_outstanding_jobs_g) {
        output_path_stats_prefix << "_con" << max_num_outstanding_jobs;
    }
    if (num_jobs_n && !num_jobs_g) {
        output_path_stats_prefix << "_n" << num_jobs;
    }
    if (start_record_num_n && !start_record_num_g) {
        output_path_stats_prefix << "_srn" << start_record_num;
    }
    if (seed_n && !seed_g) {
        output_path_stats_prefix << "_seed" << seed;
    }
    if (fairness_n && !fairness_g) {
        output_path_stats_prefix << "_fair" << fairness_threshold;
    }
    if (sched_sleep_n && !sched_sleep_g) {
        output_path_stats_prefix << "_scheds" << sched_sleep;
    }
#ifdef SUBMIT_PREGEN
    if (pregen_job_id != nullptr) {
        output_path_stats_prefix << "_job" << pregen_job_id;
    }
#endif

    std::stringstream output_path_long_prefix;
    output_path_long_prefix << output_path_stats_prefix.str();

    // Add also the one that is grouped in the output path prefix
    if (mean_inter_time_n && mean_inter_time_g) {
        output_path_long_prefix << "_iat" << mean_inter_time;
    }
    if (log_normal_sigma_n && log_normal_sigma_g) {
        output_path_long_prefix << "_lns" << log_normal_sigma;
    }
    if (max_num_outstanding_jobs_n && max_num_outstanding_jobs_g) {
        output_path_long_prefix << "_con" << max_num_outstanding_jobs;
    }
    if (num_jobs_n && num_jobs_g) {
        output_path_long_prefix << "_n" << num_jobs;
    }
    if (start_record_num_n && start_record_num_g) {
        output_path_long_prefix << "_srn" << start_record_num;
    }
    if (seed_n && seed_g) {
        output_path_long_prefix << "_seed" << seed;
    }
    if (fairness_n && fairness_g) {
        output_path_long_prefix << "_fair" << fairness_threshold;
    }
    if (sched_sleep_n && sched_sleep_g) {
        output_path_long_prefix << "_scheds" << sched_sleep;
    }

    std::vector<const char*> job_paths;
    std::vector<unsigned> job_max_outstanding_nums;
#ifdef SUBMIT_DIS
    int job_list_start = optind;
    for (unsigned i = job_list_start; i < argc; i += 3) {
        job_paths.push_back(argv[i]);
        if (i == job_list_start) {
            job_props_cum.push_back(std::stof(argv[i + 1]));
        } else {
            job_props_cum.push_back(std::stof(argv[i + 1]) + job_props_cum.back());
        }
        job_max_outstanding_nums.push_back(std::atoi(argv[i + 2]));
    }
#endif
#ifdef SUBMIT_PREGEN
    int job_list_start = optind;
    for (unsigned i = job_list_start; i < argc; i += 2) {
        job_paths.push_back(argv[i]);
        job_max_outstanding_nums.push_back(std::atoi(argv[i + 1]));
    }
#endif

    num_outstanding_jobs_per_type.resize(job_paths.size());
    latencies_per_type.resize(job_paths.size());

    std::vector<std::vector<int64_t>> input_dims;
    std::vector<std::vector<int64_t>> output_dims;

    for (const std::string& job_path : job_paths) {
        FILE* fp = fopen((job_path + ".dim").c_str(), "r");

        int d;
        std::vector<int64_t> input_dim;
        std::vector<int64_t> output_dim;
        bool is_input = true;

        while (1) {
            fscanf(fp, "%d", &d);

            if (feof(fp)) {
                break;
            }

            if (d < 0) {
                is_input = false;
                continue;
            }

            if (is_input) {
                input_dim.push_back(d);
            } else {
                output_dim.push_back(d);
            }
        }

        fclose(fp);

        input_dims.push_back(std::move(input_dim));
        output_dims.push_back(std::move(output_dim));
    }

    unsigned total_num_models = 0;
    for (unsigned job_type = 0; job_type < job_paths.size(); ++job_type) {
        total_num_models += job_max_outstanding_nums[job_type];
    }
    modules.reserve(total_num_models);

    std::vector<tvm::runtime::Module> mod_factories;
    mod_factories.reserve(job_paths.size());

    unused_modules.resize(job_paths.size());
    unused_modules_nums = new std::atomic_uint[job_paths.size()];

    for (unsigned job_type = 0; job_type < job_paths.size(); ++job_type) {
        mod_factories.push_back(tvm::runtime::Module::LoadFromFile(job_paths[job_type]));
        unused_modules[job_type].reserve(job_max_outstanding_nums[job_type]);
        for (unsigned i = 0; i < job_max_outstanding_nums[job_type]; ++i) {
            modules.emplace_back(&mod_factories.back(), input_dims[job_type], output_dims[job_type], job_type);
            unused_modules[job_type].push_back(&modules.back());
        }

        unused_modules_nums[job_type].store(job_max_outstanding_nums[job_type], std::memory_order_release);
    }

    due_jobs.resize(job_paths.size());

    unused_streams.resize(max_num_outstanding_jobs);
    for (unsigned i = 0; i < max_num_outstanding_jobs; ++i) {
        cudaStreamCreate(&unused_streams[i]);
    }

    printf("Using seed: %u\n", seed);

    std::thread monitor_thr(monitor);
    std::thread submit_thr(submit);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(monitor_thr.native_handle(), sizeof(cpu_set_t), &cpuset);

    CPU_SET(2, &cpuset);
    pthread_setaffinity_np(submit_thr.native_handle(), sizeof(cpu_set_t), &cpuset);

    monitor_thr.join();

    double time_elasped = std::chrono::duration<double, std::micro>(end_time - start_time).count();

    std::stringstream stats_output_path;
    stats_output_path << output_path_stats_prefix.str() << ".txt";
    FILE* fp = fopen(stats_output_path.str().c_str(), "a");
    if (fairness_g) {
        fprintf(fp, "%f", fairness_threshold);
    } else if (sched_sleep_g) {
        fprintf(fp, "%u", sched_sleep);
    } else { // if (mean_inter_time_g) or default
        fprintf(fp, "%d", (int)mean_inter_time);
    }
    fprintf(fp, ",%f", time_elasped);
    print_latency_stats(fp, &latencies);
    for (auto& latencies : latencies_per_type) {
        print_latency_stats(fp, &latencies);
    }
    fprintf(fp, "\n");
    fclose(fp);

    std::stringstream raw_output_path;
    raw_output_path << output_path_long_prefix.str() << "_raw.txt";
    fp = fopen(raw_output_path.str().c_str(), "w");
    for (double latency : latencies) {
        fprintf(fp, "%f\n", latency);
    }
    fclose(fp);

    std::stringstream timeline_output_path;
    timeline_output_path << output_path_long_prefix.str() << "_timeline.txt";
    fp = fopen(timeline_output_path.str().c_str(), "w");
    fprintf(fp, "%u ", num_outstanding_jobs_timeline[0]);
    for (unsigned i = 1; i < num_outstanding_jobs_timeline.size(); ++i) {
        if (i % (job_paths.size() + 1) == 0) {
            fprintf(fp, "\n");
        }
        fprintf(fp, "%u ", num_outstanding_jobs_timeline[i]);
    }
    fclose(fp);
}

