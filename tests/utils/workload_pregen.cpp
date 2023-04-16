#define DIS_LN

#include <getopt.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>

int main(int argc, char** argv) {
    double mean_inter_time;
#ifdef DIS_LN
    double log_normal_sigma = -1;
#endif
    std::vector<float> job_props_cum;
    unsigned num_jobs;
    unsigned seed;
    const char* pregen_prefix;

    while (true) {
        static struct option long_options[] = {
            {"iat",                required_argument, 0, 'i'},
#ifdef DIS_LN
            {"ln_sigma",           required_argument, 0, 'l'},
#endif
            {"num_jobs",           required_argument, 0, 'n'},
            {"seed",               required_argument, 0, 's'},
            {"pregen_prefix",      required_argument, 0, 'g'},
            {0,                    0,                 0,  0 }
        };

        int opt_idx = 0;
        int opt_val = getopt_long(argc, argv, "i:l:n:s:g:", long_options, &opt_idx);

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
            case 'n':
                num_jobs = atoi(optarg);
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'g':
                pregen_prefix = optarg;
                break;
        }
    }

    int job_list_start = optind;
    for (unsigned i = job_list_start; i < argc; ++i) {
        if (i == job_list_start) {
            job_props_cum.push_back(std::stof(argv[i]));
        } else {
            job_props_cum.push_back(std::stof(argv[i]) + job_props_cum.back());
        }
    }

    static std::mt19937 gen(seed);
    static std::uniform_real_distribution<float> d_type(0, 1);
#ifdef DIS_EXP
    static std::exponential_distribution<> d_inter(1. / mean_inter_time);
#else // if DIS_LN
    static const double log_normal_mu = log(mean_inter_time) - log_normal_sigma * log_normal_sigma / 2;
    static std::lognormal_distribution<> d_inter(log_normal_mu, log_normal_sigma);
#endif

    std::stringstream pregen_path;
    pregen_path << pregen_prefix;
    pregen_path << "_iat" << mean_inter_time;
    pregen_path << "_lns" << log_normal_sigma;
    pregen_path << "_n" << num_jobs;
    pregen_path << "_seed" << seed;

    std::stringstream pregen_all_path;
    pregen_all_path << pregen_path.str() << ".txt";
    FILE* fp = fopen(pregen_all_path.str().c_str(), "w");

    std::vector<FILE*> fps;
    for (unsigned i = 0; i < job_props_cum.size(); ++i) {
        std::stringstream pregen_splitted_path;
        pregen_splitted_path << pregen_path.str();
        pregen_splitted_path << "_job" << i << ".txt";
        fps.push_back(fopen(pregen_splitted_path.str().c_str(), "w"));
    }

    double next_submit_time = 0;
    for (unsigned i = 0; i < num_jobs; ++i) {
        unsigned job_type;
        job_type = std::lower_bound(job_props_cum.begin(), job_props_cum.end(), d_type(gen)) - job_props_cum.begin();

        double next_inter = d_inter(gen);

        fprintf(fp, "%u %f\n", job_type, next_submit_time);

        fprintf(fps[job_type], "0 %f\n", next_submit_time);

        next_submit_time += next_inter;
    }

    for (FILE* fp : fps) {
        fclose(fp);
    }
    fclose(fp);
}

