# Goal

We aim to demonstrate head of line blocking due to poor utilization of GPU hardware queues.
Specifically, the GPU block-to-SM dispatcher will not dequeue a kernel at the head of a queue if a dependent kernel is already executing (in any SM).
This means that, theoretically, it is possible for the heads of all the HW queues of the GPU to host a non-schedulable kernel. If there are schedulable kernels later in the queue, they will be head-of-line blocked. Further, if there was more real estate in the GPU to process kernels, this means we are under-utilizing the device.

# Experiment setup

Hardware
---
Tesla T4 Turing (see https://docs.nvidia.com/cuda/turing-tuning-guide/index.html):

```
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Tesla T4"
  CUDA Driver Version / Runtime Version          11.7 / 11.6
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 14972 MBytes (15699148800 bytes)
  (040) Multiprocessors, (064) CUDA Cores/MP:    2560 CUDA Cores
  GPU Max Clock rate:                            1590 MHz (1.59 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 59 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.7, CUDA Runtime Version = 11.6, NumDevs = 1
Result = PASS
```

Synthetic load: job anatomy
---
We implement a job as a sequence of kernels scheduled on the same CUDA stream. We use 1 stream per job and never reuse a stream.

- 8 kernels per job, 128 threads per kernel
- Theoretical max concurrency per SM = 1024 / 128 = 8 kernels
- Theoretical max concurrency on the GPU = 8 * 40 = 320 jobs

Synthetic load: theoretical concurrency
---
- kernel runtime: 316us. expected JCT 316us * 8 = 2528us
    * sustainable job/s for 1 job: 1e6 / 2528 = 395 jobs/s
    * sustainable job/s for the entire GPU: 395*320 = 126400 jobs/s

- kernel runtime: 627us. expected JCT 627us * 8 = 5016us
    * sustainable job/s for 1 job: 1e6 / 5016 = 199 jobs/s
    * sustainable job/s for the entire GPU: 199*320 = 63795 jobs/s

- kernel runtime: 995us. expected JCT 995us * 8 = 7960us
    * sustainable job/s for 1 job: 1e6 / 7960 = 125 jobs/s
    * sustainable job/s for the entire GPU: 125*320 = 40000 jobs/s

- kernel runtime: 3967us. expected JCT 3967 * 8 = 31736us
    * sustainable job/s for 1 job: 1e6 / 31736 = 31 jobs/s
    * sustainable job/s for the entire GPU: 31*320 = 9920 jobs/s

# Methodology

We compare two scheduling policy: one where we schedule at once all kernels for new jobs ("FIFO-full") and one where we schedule kernels one at a time, moving through available jobs in a round robin fashion ("FIFO-kernel-RR"). Given our GPU has 32 HW queues and the entire GPU can theoretically host 320 jobs, if there are more than 32 jobs' kernels in the HW queues the FIFO policy will allow kernels to be blocked in queue. In both cases we use uniform arrival distributions.

FIFO-KERNEL-RR: pseudo code
---

```
low := 0
high := 0
kernels = [] # Maintain a list of size NJOBS, initialized at 0, storing the current kernel a job has to run next
start_times = [] # Record jobs' starting time

while low < NJOBS:
    # Is it time to account for one more job?
    if (next job arrived):
        start_times[high] = now()
        next_arrival = now() + interval
        high += 1

    # Nothing to do
    if (high == low)
        continue

    # Iterate over all running jobs
    for (job_id = low, job_id < high, ++job_id):
        # Check which kernel job_id has to run. Increment the sent kernel count.
        kernel = kernels[job_id]++

        # If this is this job's last kernel, increment the low watermark
        if (kernels[job_id] == KERNEL_PER_JOB):
            low += 1

        # Run the kernel in the job's stream
        launch_kernel<<<1, 128, 0, streams[job_id]>>>
```

Note that:
- When the loop starts a kernel for a job, it doesn't know whether the previous kernel for this job is done (i.e. if the stream is empty). This means if a later job finishes first, we would increment low and potentially skip kernels for an earlier job. This seems to never happen, though has we have a validating loop to check all jobs( kernels have been scheduled and so far never failed.
- When (high - low) > 32, it means we are processing more jobs then there are HQ queues. In this situation we should be able to process more jobs at once than FIFO-full.
- Conversely, when (high - low) lt 32, we should do no better than FIFO-full and maybe sligthly worse, as we now introce delay between each kernel's dispatching

# Implementation details

- 1 sending thread
- 1 gathering thread
- threads are pinned
- CUDA_DEVICE_MAX_CONNECTIONS=32

# Current observations

I am observing two issues:
- The expected "better" dispatching mode (round robin across kernels) performs worse than FIFO (submit all kernels at once)
- For both dispatching mode, the latency increases linearly as we submit jobs. This increase tendency starts at a very low load, i.e. it is always present.
    * Can it be something unrelated to arrival rates?
- For FIFO-kernel-RR, I am observing some queue: at 5k rps, qlen = 2 and at 10k rps, qlen = 7. qlen = high - low.
    * Ideally we would be able to measure HW queues occupancy

- I am noticing some variability in JCT: even at low load for the same kernel runtime of 1ms I have had 19ms and 9ms completion time. It could be because turbo boost kicks in...


Changing the number of streams
---
- Tried 32,64,128, 1 per job.
- More streams is always better in terms of latency, regardless of the scheduling strategy. Best is 1 per job.
- With longer kernels (1ms to 4ms), FIFO-full benefits significantly more from increasing the number of streams than FIFO-kernel-RR.

My explanation is that tweaking the number of stream is the best example of reducing HoL blocking. It's not that interesting for us because one can use streams w/o allis.
Decreasing the number of streams means more concurrent jobs land on the same HW queues (pigeon hole pcpl). When number of streams = number of HW queues, we arrive to the exact situation figure 2 wants to demonstrate. The optimal situation for this experiment should be 1 stream per job.

Increasing kernel runtimes
---
- Did not change the linear latency increase pattern.
- As expected, overload happens faster with longer kernels, for a given sending rate. E.g., about 2.5x faster for 4ms/kernel from 1ms/kernel at 2000rps vs 5000rps.
-> both these tend to indicate that the issue is not with kernel size, but rather dispatching method.


# Vrac
- Could we take advantage from parallelizing the kernels of a given job? We don't consider kernels //ism afaik

- Try building distribution of inter-kernel scheduling times. Should show the overheads introduced by FIFO-kernel-RR
- use nvprof
- delay introduce between kernels by FIFO-kernel-raw
