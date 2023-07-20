#define DIS_LN

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <queue>
#include <random>
#include <getopt.h>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>

using namespace std::chrono_literals;

std::queue<std::pair<unsigned, double>> sched;
std::mutex sched_mtx;
std::chrono::time_point<std::chrono::steady_clock> start_time;
std::vector<std::vector<double>> latencies_per_thr;

class MyModule {
  public:
    MyModule(tvm::runtime::Module* mod_factory, const std::vector<int64_t>& input_dim, const std::vector<int64_t>& output_dim) {
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
    }

    void run() {
        input_dev_.CopyFrom(input_);

        set_input_(0, input_dev_);
        run_();
        get_output_(0, output_dev_);

        output_.CopyFrom(output_dev_);
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
};

void worker(std::vector<MyModule>* models, unsigned thr_id) {
    TVMStreamHandle stream;
    TVMStreamCreate(kDLCUDA, 0, &stream);
    TVMSetStream(kDLCUDA, 0, stream);

    while (1) {
        std::unique_lock<std::mutex> lk(sched_mtx);
        if (sched.empty()) {
            break;
        }

        auto next_run = sched.front();
        sched.pop();
        lk.unlock();

        while (1) {
            auto cur_time = std::chrono::steady_clock::now();
            auto time_elasped = std::chrono::duration<double, std::micro>(cur_time - start_time).count();

            if (time_elasped >= next_run.second) {
                (*models)[next_run.first].run();
                //TVMSynchronize(kDLCUDA, 0, stream);
                auto after_time = std::chrono::steady_clock::now();

                auto latency = std::chrono::duration<double, std::micro>(after_time - start_time).count() - next_run.second;
                latencies_per_thr[thr_id].push_back(latency);

                break;
            }
        }
    }
}

int main(int argc, char** argv) {
    double mean_inter_time = -1;
#ifdef DIS_LN
    double log_normal_sigma = -1;
#endif
    unsigned num_jobs = -1;
    unsigned seed = -1;
    unsigned num_thrs = -1;
    std::string output_path;

    while (true) {
        static struct option long_options[] = {
            {"iat",                required_argument, 0, 'i'},
#ifdef DIS_LN
            {"ln_sigma",           required_argument, 0, 'l'},
#endif
            {"concurrency",        required_argument, 0, 'c'},
            {"num_jobs",           required_argument, 0, 'n'},
            {"seed",               required_argument, 0, 's'},
            {"output_path",        required_argument, 0, 'o'},
 
            {0,                    0,                 0,  0 }
        };

        int opt_idx = 0;
        int opt_val = getopt_long(argc, argv, "i:l:c:n:s:", long_options, &opt_idx);

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
                num_thrs = atoi(optarg);
                break;
            case 'n':
                num_jobs = atoi(optarg);
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'o':
                output_path = optarg;
                break;
        }
    }

    std::vector<std::string> model_paths;
    std::vector<float> job_props_cum;

    for (int i = optind; i < argc; i += 2) {
        model_paths.push_back(argv[i]);

        if (i == optind) {
            job_props_cum.push_back(std::stof(argv[i + 1]));
        } else {
            job_props_cum.push_back(std::stof(argv[i + 1]) + job_props_cum.back());
        }
    }

    std::vector<std::vector<int64_t>> input_dims;
    std::vector<std::vector<int64_t>> output_dims;

    for (const std::string& model_path : model_paths) {
        FILE* fp = fopen((model_path + ".dim").c_str(), "r");

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

    std::vector<std::vector<MyModule>> models;
    std::vector<tvm::runtime::Module> mod_factories;
    models.resize(num_thrs);
    mod_factories.reserve(model_paths.size() * num_thrs);

    for (unsigned i = 0; i < num_thrs; ++i) {
        models[i].reserve(model_paths.size());
    }

    for (unsigned i = 0; i < model_paths.size(); ++i) {
        for (unsigned j = 0; j < num_thrs; ++j) {
            mod_factories.push_back(tvm::runtime::Module::LoadFromFile(model_paths[i]));
            models[j].emplace_back(&mod_factories.back(), input_dims[i], output_dims[i]);
        }
    }

    std::random_device rd;
    std::mt19937 gen(seed);
#ifdef DIS_EXP
    std::exponential_distribution<> d_inter(1. / mean_inter_time);
#else // if DIS_LN
    double log_normal_mu = log(mean_inter_time) - log_normal_sigma * log_normal_sigma / 2;
    std::lognormal_distribution<> d_inter(log_normal_mu, log_normal_sigma);
#endif
    std::uniform_real_distribution<float> d_type(0, 1);

    double next_submit_time = 0;
    for (unsigned i = 0; i < num_jobs; ++i) {
        sched.push(std::make_pair(std::lower_bound(job_props_cum.begin(), job_props_cum.end(), d_type(gen)) - job_props_cum.begin(), next_submit_time));
        next_submit_time += d_inter(gen);
    }

    latencies_per_thr.resize(num_thrs);

    start_time = std::chrono::steady_clock::now() + 2s;

    std::vector<std::thread> thrs;
    for (unsigned i = 0; i < num_thrs; ++i) {
        thrs.emplace_back(worker, &(models[i]), i);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i * 2 + 1, &cpuset);
        pthread_setaffinity_np(thrs.back().native_handle(), sizeof(cpu_set_t), &cpuset);
    }
    for (auto& thr : thrs) {
        thr.join();
    }

    auto end_time = std::chrono::steady_clock::now();
    double time_elasped = std::chrono::duration<double, std::micro>(end_time - start_time).count();
    
    std::vector<double> latencies;
    for (unsigned i = 0; i < num_thrs; ++i) {
        latencies.insert(latencies.end(), latencies_per_thr[i].begin(), latencies_per_thr[i].end());
    }

    std::sort(latencies.begin(), latencies.end());

    double mean = 0;
    double mean_sqr = 0;

    for (double latency : latencies) {
        mean += (latency / (double)latencies.size());
        mean_sqr += latency * latency / (double)latencies.size();
    }

    double sd = sqrt((mean_sqr - mean * mean) * ((double)latencies.size() / ((double)latencies.size() - 1)));

    double p50 = latencies[latencies.size() / 2];
    double p90 = latencies[latencies.size() * 0.90];
    double p95 = latencies[latencies.size() * 0.95];
    double p99 = latencies[latencies.size() * 0.99];
    double max = *std::max_element(latencies.begin(), latencies.end());

    FILE* fp = fopen(output_path.c_str(), "a");
    fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f\n", mean_inter_time, time_elasped, mean, p50, p90, p95, p99, max, sd);
    fclose(fp);
}

