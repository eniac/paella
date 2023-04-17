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
#include <map>
#include <queue>

#define cudaStreamNonBlocking 0x01

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
#define NJOBS 10000
#define NSTREAMS NJOBS
#define KERNEL_PER_JOB 8
#define THREADS_PER_KERNEL 1024

void jct_gatherer(int *kernels_completed, uint64_t *jct, uint64_t *starts) {
    std::unordered_map<int, int> done;
    done.reserve(NJOBS);
    for (int i = 0; i < NJOBS; ++i) {
        done[i] = 0;
    }
    uint64_t completed = 0;
    while (completed < NJOBS) {
        for (int i = 0; i < NJOBS; ++i) {
            if (kernels_completed[i] == KERNEL_PER_JOB && !done[i]) {
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
   On dedos09/ T4
   ---
   with -Xptxas -O0, saxpy, 128 threads/kernel:
    - 5500: median 315us
    - 11000: median 627us
    - 15000: median 853us
    - 17500: median 995us
    - 70000: median 3967us

   with -03, factorial:
   - 120000: 1ms

   with -03 -arch=sm_75, factorial, 128 threads/kernel:
   - 120000: 995us
   - 157080: 1307us
   - 1200000: 9989us

   with -03 -arch=sm_75, factorial, 1024 threads/kernel:
   - 25000 1015us
   - 125000 5061us
   - 250000 10015us
   - 600000: 24ms
   - 1200000: 48.5ms
*/
#define NLOOPS 125000
/*
__global__ void saxpy(volatile int job_id, volatile int kernel_id, volatile int *kernels_completed) {
    volatile int i = 0;
    for (i = 0; i++ < NLOOPS; ++i) {
        i += blockIdx.x * blockDim.x + threadIdx.x; // thread index
    }
    if (threadIdx.x == 0 && kernel_id == KERNEL_PER_JOB - 1) {
        kernels_completed[job_id] = 1;
    }
}
*/

__global__ void factorial(int job_id, int kernel_id,
                          unsigned num_loops,
                          uint32_t *scratchpad, int *kernels_completed) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 1;
    for (unsigned i = 1; i <= num_loops; ++i) {
        tmp *= i;
    }
    if (threadIdx.x == 0) { // && kernel_id == KERNEL_PER_JOB - 1) {
        scratchpad[id] = tmp;
        kernels_completed[job_id] = kernel_id + 1;
    }
}

#define NSAMPLES 1000
void measure_kernel_runtime(int *kernels_completed, uint32_t *kernels_scratchpad) {
    std::cout << "Measuring kernel runtime & dispatch time" << std::endl;
    std::cout << "number of loops in kernel: " << NLOOPS << std::endl;
    uint64_t measure_start = rdtscp(NULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> runtimes;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < NSAMPLES; ++i) {
        cudaError_t ret = cudaEventRecord(start, stream);
        if (ret != cudaSuccess) {
            std::cerr << "Error during cudaEventRecord: " << cudaGetErrorString(ret) << std::endl;
        }

        factorial<<<1, THREADS_PER_KERNEL, 0, stream>>>(i, i % KERNEL_PER_JOB, NLOOPS, kernels_scratchpad, kernels_completed);
        ret = cudaEventRecord(stop, stream);
        if (ret != cudaSuccess) {
            std::cerr << "Error during cudaEventRecord: " << cudaGetErrorString(ret) << std::endl;
        }

        ret = cudaEventSynchronize(stop);
        if (ret != cudaSuccess) {
            std::cerr << "Error during cudaEventSynchronize: " << cudaGetErrorString(ret) << std::endl;
        }
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

    int n_hwqs = 0;
    if (const char* max_cuda_conns = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS")) {
        std::cout << "CUDA_DEVICE_MAX_CONNECTIONS is: " << max_cuda_conns << std::endl;
        n_hwqs = std::atoi(max_cuda_conns);
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
        cudaError_t ret = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        if (ret != cudaSuccess) {
            std::cerr << "Error during cudaStreamCreateWithFlags: " << cudaGetErrorString(ret) << std::endl;
        }
    }

    // allocate array of kernel completion kernels_completeds on pinned memory
    int *kernels_completed;
    cudaError_t ret = cudaMallocHost(&kernels_completed, NJOBS * sizeof(int));
    if (ret != 0) {
        fprintf(stderr, "cudaMallocHost error: %d\n", ret);
    }

    uint32_t *kernels_scratchpad;
    ret = cudaMalloc(&kernels_scratchpad, NJOBS * sizeof(uint32_t));
    if (ret != 0) {
        fprintf(stderr, "cudaMalloc error: %d\n", ret);
    }

    if (mode == 0) {
        measure_kernel_runtime(kernels_completed, kernels_scratchpad);
        exit(1);
    }

    // Allocate arrays to store timing information, track kernels sent
    uint64_t *jct = (uint64_t *) malloc(NJOBS * sizeof(uint64_t));
    uint64_t *starts = (uint64_t *) malloc(NJOBS * sizeof(uint64_t));
    std::thread gatherer(jct_gatherer, kernels_completed, jct, starts);
    pin_thread(gatherer.native_handle(), 4);

    // Allocate an array to track queue length at the server. We will record qlen every time a new request arrive.
    // This is only used for "FIFO-kernel-RR"
    int *qlens = (int *) malloc(NJOBS * sizeof(int));

    uint64_t t0 = rdtscp(NULL);

    if (mode == 1) {
        // Dummy mode (all kernels at once)
        std::cout << "Sending kernels in dummy mode" << std::endl;

        struct timespec sleeptime;
        sleeptime.tv_sec = 0;
        sleeptime.tv_nsec = interval_ns;
        uint64_t interval_in_cycles = interval_ns * cycles_per_ns;
        uint64_t last_send_cycle = 0;
        int i = 0;
        while (i < NJOBS) {
            // Check if we are due to send a new job
            uint64_t now = rdtscp(NULL);
            if ((now - last_send_cycle) > interval_in_cycles) {
                // It takes about 22us to do this dispatch
                starts[i] = now;
                for (int j = 0; j < KERNEL_PER_JOB; ++j) {
                     factorial<<<1, THREADS_PER_KERNEL, 0, streams[i % NSTREAMS]>>>(i, j, NLOOPS, kernels_scratchpad, kernels_completed);
                     cudaError_t err = cudaGetLastError();
                     if (err != 0) {
                         std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
                     }
                }
                last_send_cycle = now;
                i++;
            }
        }

    } else if (mode == 2) {
        // Better mode (one kernel at a time, across max theoretical concurrency)
        std::cout << "Sending kernels in better mode" << std::endl;
        uint64_t interval = static_cast<uint64_t>(interval_ns * cycles_per_ns);
        std::queue<std::pair<int,int>> inflight_jobs;
        uint64_t now = rdtscp(NULL);
        uint64_t next_arrival = now + interval;
        int dispatching = 0;
        int sent = 0;
        while (sent < NJOBS) {
            now = rdtscp(NULL);
            if (now >= next_arrival && dispatching < NJOBS) {
                // Record start time
                starts[dispatching] = now;
                // Enqueue new job
                inflight_jobs.push(std::pair<int,int>(dispatching, 0));
                // Set next arrival time
                next_arrival = now + interval;
                // Record queue length
                qlens[sent] = inflight_jobs.size();

                dispatching++;
            }

            size_t n = inflight_jobs.size();
            for (size_t i = 0; i < n; ++i) {
                // Get a job to check
                auto job = inflight_jobs.front();
                int job_id = job.first;
                int kernel_sent = job.second;
                inflight_jobs.pop();

                // If the job hasn't started or has completed the last kernel it launched, launch the next one
                if ((kernel_sent == 0) || (kernels_completed[job_id] == kernel_sent)) {
                    // Launch a kernel
                    // std::cout << "dispatching kernel " << kernel_sent << " for job :" << job_id << std::endl;
                    factorial<<<1, THREADS_PER_KERNEL, 0, streams[job_id % NSTREAMS]>>>(job_id, kernel_sent, NLOOPS, kernels_scratchpad, kernels_completed);
                    cudaError_t err = cudaGetLastError();
                    if (err != 0) {
                        std::cerr << "Error launching kernel: " << cudaGetErrorString(err) << std::endl;
                    }

                    // If the job still has kernels to send, enqueue it back
                    if (kernel_sent + 1 < KERNEL_PER_JOB) {
                        inflight_jobs.push(std::pair<int,int>(job_id, kernel_sent + 1));
                    } else {
                        sent++;
                    }
                } else {
                    // If not, re-enqueue and move to next job
                    inflight_jobs.push(std::pair<int,int>(job_id, kernel_sent));
                }
            }
        }
    }

    uint64_t t1 = rdtscp(NULL);
    uint64_t runtime = ((t1 - t0) / cycles_per_us);
    std::cout << "Done sending kernels in " << runtime << " us" << std::endl;

    // cudaDeviceSynchronize(); // We don't really need to synchronize, and synchronize generate poll syscalls.
    gatherer.join();

    for (int i = 0; i < NJOBS; ++i) {
        if (kernels_completed[i] < KERNEL_PER_JOB) {
            std::cerr << "job " << i << " sent only " << kernels_completed[i] << std::endl;
        }
        assert(kernels_completed[i] == KERNEL_PER_JOB);
    }

    std::string experiment_label(argv[2]);

    // Job latencies results
    std::string fname = experiment_label + "-" + std::to_string(int(sending_rate)) + "-results.csv";
    std::ofstream results(fname);
    std::cout << "Formatting results in " << fname << std::endl;
    results << "JOB_ID\tJCT\tRATE\tDURATION\tKERNEL_PER_JOB\tKERNEL_LOOPS\tNJOBS\tNSTREAMS\tNHWQS" << std::endl;

    // Qlen results
    std::string qlen_fname = experiment_label + "-" + std::to_string(int(sending_rate)) + "-qlen-results.csv";
    std::ofstream qlen_results(qlen_fname);
    std::cout << "Formatting qlen results in " << qlen_fname << std::endl;
    qlen_results << "TIME\tQLEN\tRATE" << std::endl;

    for (int i = 0; i < NJOBS; ++i) {
        // JCT
        results << std::fixed << std::setprecision(0)
                << i << "\t"
                << jct[i] / cycles_per_us << "\t"
                << sending_rate << "\t"
                << runtime << "\t" // us
                << KERNEL_PER_JOB << "\t"
                << NLOOPS << "\t"
                << NJOBS << "\t"
                << NSTREAMS << "\t"
                << n_hwqs << std::endl;

        // QLEN
        // Offset each timestamp with the smallest start time, and record time in seconds
        qlen_results << (starts[i] - starts[0]) / (cycles_per_us * 1e6) << "\t"
                     << qlens[i] << "\t"
                     << sending_rate << std::endl;
    }

    return 0;
}
