#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <unordered_map>
#include <iomanip>

using namespace std::chrono_literals;

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

#define NQUEUES_TO_USE 32
#define NJOBS 10
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
                done[i] = 1;
                jct[i] = rdtscp(NULL) - starts[i];
                completed++;
            }
        }
    }
    std::this_thread::sleep_for(2000ms);
}

#define NLOOPS 5500
__global__ void saxpy(volatile int job_id, volatile int kernel_id, volatile int *signal) {
    volatile int i = 0;
    for (i = 0; i++ < NLOOPS; ++i) {
        i += blockIdx.x * blockDim.x + threadIdx.x; // thread index
    }
    if (threadIdx.x == 0 && kernel_id == KERNEL_PER_JOB - 1) {
        signal[job_id] = 1;
    }
}

#define NSAMPLES 100000
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

    int sleep_time_ns = atoi(argv[1]);
    std::cout << "sleep time: " << sleep_time_ns << " ns" << std::endl;

    float sending_rate = 1e9 / sleep_time_ns;
    std::cout << "sending rate: " << sending_rate << " jobs per seconds " << std::endl;

    // Control sending rate
    struct timespec sleeptime;
    sleeptime.tv_sec = 0;
    sleeptime.tv_nsec = sleep_time_ns;

    // How many streams to use
    std::vector<cudaStream_t> streams;
    streams.resize(NQUEUES_TO_USE);
    for (int i = 0; i < streams.size(); ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // allocate array of kernel completion signals on pinned memory
    int *signal;
    cudaError_t ret = cudaMallocHost(&signal, NJOBS * sizeof(int));
    if (ret != 0) {
        fprintf(stderr, "cudaMallocHost error: %d\n", ret);
    }

    // measure kernel runtimes
    measure_kernel_runtime(signal);

    uint64_t *jct = (uint64_t *) malloc(NJOBS * sizeof(uint64_t));
    uint64_t *starts = (uint64_t *) malloc(NJOBS * sizeof(uint64_t));
    std::thread gatherer(jct_gatherer, signal, jct, starts);

    std::cout << "Sending kernels in dummy mode" << std::endl;
    // Dummy mode (all kernels at once)
    for (int i = 0; i < NJOBS; ++i) {
        starts[i] = rdtscp(NULL);
        for (int j = 0; j < KERNEL_PER_JOB; ++j) {
            saxpy<<<1, 128, 0, streams[i % streams.size()]>>>(i, j, signal);
        }
        nanosleep(&sleeptime, NULL);
    }
    std::cout << "Done sending kernels" << std::endl;

    /*
    // Better mode (one kernel at a time, across max theoretical concurrency)
    int sent = 0;
    int low = 0;
    int high = low + 40*KERNEL_PER_JOB;
    while (sent < NJOBS) {
        for (int j = 0; j < KERNEL_PER_JOB; ++j) {
            for (int i = low; i < high; ++i) {
                if (j == 0) {
                    starts[i] = rdtscp(NULL);
                }
                saxpy<<<1, 128, 0, streams[i % streams.size()]>>>(i, j, signal);
            }
            low = high;
            high = low + 40*KERNEL_PER_JOB;
        }
        sent += high - low;
    }
     */

    gatherer.join();

    std::string fname = std::to_string(int(sending_rate)) + "-results.csv";
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
