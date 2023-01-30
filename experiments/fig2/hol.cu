#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <thread>
#include <pthread.h>
#include <vector>
#include <unordered_map>
#include <iomanip>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <cstdlib>

using namespace std::chrono_literals;

static inline void pin_thread(pthread_t thread, u_int16_t cpu) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "could not pin thread: " << std::strerror(errno) << std::endl;
        exit(1);
    }
}

static inline uint64_t rdtscp(uint32_t *auxp) {
    uint32_t a, d, c;
    asm volatile("rdtscp" : "=a" (a), "=d" (d), "=c" (c));
    if (auxp)
        *auxp = c;
    return ((uint64_t)a) | (((uint64_t)d) << 32);
}

static inline void cpu_serialize(void) {
        asm volatile("xorl %%eax, %%eax\n\t"
             "cpuid" : : : "%rax", "%rbx", "%rcx", "%rdx");
}

float cycles_per_ns;
uint32_t cycles_per_us;
// From Adam's base OS
inline int time_calibrate_tsc(void) {
    struct timespec sleeptime;
    sleeptime.tv_sec = 0;
    sleeptime.tv_nsec = 5E8; /* 1/2 second */
    struct timespec t_start, t_end;

    cpu_serialize();
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t_start) == 0) {
        uint64_t ns, end, start;

        start = rdtscp(NULL);
        nanosleep(&sleeptime, NULL);
        clock_gettime(CLOCK_MONOTONIC_RAW, &t_end);
        end = rdtscp(NULL);
        ns = ((t_end.tv_sec - t_start.tv_sec) * 1E9);
        ns += (t_end.tv_nsec - t_start.tv_nsec);

        cycles_per_ns = ((end - start) * 1.0) / ns;
        printf("Time calibration: detected %.03f ticks / ns\n", cycles_per_ns);
        cycles_per_us = cycles_per_ns * 1e3;

        return 0;
    }

    return -1;
}

#define NSMS 40
#define N_HW_QUEUES 32
#define NJOBS 100000
#define NSTREAMS NJOBS
#define KERNEL_PER_JOB 8
#define THREADS_PER_KERNEL 128

void jct_gatherer(int *signal, uint64_t *jct, uint64_t *starts) {
    std::unordered_map<int, int> done;
    done.reserve(NJOBS);
    for (int i = 0; i < NJOBS; ++i) {
        done[i] = false;
    }
    uint64_t completed = 0;
    while (completed < NJOBS) {
        for (int i = 0; i < NJOBS; ++i) {
            if (signal[i] && !done[i]) {
                // std::cout << "job " << i << " completed" << std::endl;
                done[i] = 1;
                jct[i] = rdtscp(NULL) - starts[i];
                completed++;
            }
        }
    }
    std::this_thread::sleep_for(2000ms);
}

/*
   On dedos09/ T4:
    - 5500: median 315us
    - 11000: median 627us
    - 15000: median 853us
    - 17500: median 995us
    - 70000: median 3967us
*/
#define NLOOPS 17500
__global__ void saxpy(volatile int job_id, volatile int kernel_id, volatile int *signal) {
    volatile int i = 0;
    for (i = 0; i++ < NLOOPS; ++i) {
        i += blockIdx.x * blockDim.x + threadIdx.x; // thread index
    }
    if (threadIdx.x == 0 && kernel_id == KERNEL_PER_JOB - 1) {
        signal[job_id] = 1;
    }
}

#define NSAMPLES 10000
void measure_kernel_runtime(int *signal) {
    uint64_t measure_start = rdtscp(NULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> runtimes;

    for (int i = 0; i < NSAMPLES; ++i) {
        cudaEventRecord(start);
        saxpy<<<1, 128>>>(0, 0, signal);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float kernel_runtime_in_ms = 0;
        cudaEventElapsedTime(&kernel_runtime_in_ms, start, stop);
        runtimes.push_back(kernel_runtime_in_ms);
    }

    uint64_t measure_end = rdtscp(NULL);
    std::cout << "measured kernel runtime in "
              << (measure_end - measure_start) / cycles_per_us
              << " us"
              << std::endl;

    std::sort(runtimes.begin(), runtimes.end());
    float median = runtimes[NSAMPLES * .5];
    float p99 = runtimes[NSAMPLES * .99];
    std::cout << "synthetic kernel runtimes in us (over " << NSAMPLES << " samples)" << std::endl;
    std::cout << "p50: " << median * 1e3 << std::endl;
    std::cout << "p99: " << p99 * 1e3 << std::endl;
}

int main(int argc, char** argv) {
    time_calibrate_tsc();

    pthread_t current_thread = pthread_self();
    pin_thread(current_thread, 2);

    if (const char* max_cuda_conns = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS")) {
        std::cout << "CUDA_DEVICE_MAX_CONNECTIONS is: " << max_cuda_conns << std::endl;
    } else {
        std::cout << "WARNING: CUDA_DEVICE_MAX_CONNECTIONS is not set!!" << std::endl;
    }

    /*
        0: MEASURE KERNEL RUNTIME
        1: DUMMY
        2: BETTER
    */
    int mode = atoi(argv[1]);

    int interval_ns = atoi(argv[3]);
    std::cout << "sending interval: " << interval_ns << " ns" << std::endl;

    float sending_rate = 1e9 / interval_ns;
    std::cout << "sending rate: " << sending_rate << " jobs per seconds " << std::endl;

    // Use one stream per job
    std::cout << "Creating " << NSTREAMS << " streams" << std::endl;
    std::vector<cudaStream_t> streams;
    streams.resize(NSTREAMS);
    for (int i = 0; i < streams.size(); ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // allocate array of kernel completion signals on pinned memory
    int *signal;
    cudaError_t ret = cudaMallocHost(&signal, NJOBS * sizeof(int));
    if (ret != 0) {
        fprintf(stderr, "cudaMallocHost error: %d\n", ret);
    }

    if (mode == 0) {
        std::cout << "Measuring kernel runtime & dispatch time" << std::endl;
        measure_kernel_runtime(signal);
        exit(1);
    }

    uint64_t *jct = (uint64_t *) malloc(NJOBS * sizeof(uint64_t));
    uint64_t *starts = (uint64_t *) malloc(NJOBS * sizeof(uint64_t));
    std::thread gatherer(jct_gatherer, signal, jct, starts);
    pin_thread(gatherer.native_handle(), 4);

    uint32_t *jobs_sent_kernels = (uint32_t *) calloc(NJOBS, sizeof(uint32_t));

    uint64_t t0 = rdtscp(NULL);

    if (mode == 1) {
        // Dummy mode (all kernels at once)
        std::cout << "Sending kernels in dummy mode" << std::endl;

        struct timespec sleeptime;
        sleeptime.tv_sec = 0;
        sleeptime.tv_nsec = interval_ns;

        for (int i = 0; i < NJOBS; ++i) {
            // XXX Assumes the time to execute the next 4 lines < sleeptime
            starts[i] = rdtscp(NULL);
            for (int j = 0; j < KERNEL_PER_JOB; ++j) {
                 saxpy<<<1, 128, 0, streams[i % NSTREAMS]>>>(i, j, signal);
                 cudaError_t err = cudaGetLastError();
                 if (err != 0) {
                     std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
                 }
            }
            nanosleep(&sleeptime, NULL);
        }
    } else if (mode == 2) {
        // Better mode (one kernel at a time, across max theoretical concurrency)
        std::cout << "Sending kernels in better mode" << std::endl;
        uint64_t interval = static_cast<uint64_t>(interval_ns * cycles_per_ns);
        int job_low = 0;
        int job_high = 0;
        uint64_t now = rdtscp(NULL);
        uint64_t next_arrival = now + interval;
        while (job_low < NJOBS) {
            now = rdtscp(NULL);
            if (now >= next_arrival && job_high < NJOBS) {
                starts[job_high++] = now;
                next_arrival = now + interval;
            }

            if (job_low == job_high) {
                continue;
            }

            for (int i = job_low; i < job_high; ++i) {
                uint32_t kernel = jobs_sent_kernels[i];
                if (++jobs_sent_kernels[i] == KERNEL_PER_JOB) {
                    job_low += 1;
                }
                saxpy<<<1, 128, 0, streams[i % NSTREAMS]>>>(i, kernel, signal);
                cudaError_t err = cudaGetLastError();
                if (err != 0) {
                    std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
                }
            }
            /*
            std::cout << "Total jobs dispatched so far: " << job_low << std::endl;
            std::cout << "Jobs in flight: " << job_high - job_low << std::endl;
            */
        }

        for (int i = 0; i < NJOBS; ++i) {
            assert(jobs_sent_kernels[i] == KERNEL_PER_JOB);
        }
    }

    uint64_t t1 = rdtscp(NULL);
    uint64_t runtime = ((t1 - t0) / cycles_per_us);
    std::cout << "Done sending kernels in " << runtime << " us" << std::endl;

    cudaDeviceSynchronize();
    gatherer.join();

    std::string experiment_label(argv[2]);
    std::string fname = experiment_label + "-" + std::to_string(int(sending_rate)) + "-results.csv";
    std::ofstream results(fname);
    std::cout << "Formatting results in " << fname << std::endl;
    results << "JOB_ID\tJCT\tRATE" << std::endl;
    for (int i = 0; i < NJOBS; ++i) {
        results << std::fixed << std::setprecision(0)
                << i << "\t"
                << jct[i] / cycles_per_us << "\t"
                << sending_rate << std::endl;
    }

    return 0;
}
