#include <llis/job.h>

#include <cstdio>

__global__ void helloworld(void* job, llis::ipc::ShmChannelGpu gpu2sched_channel) {
    printf("Hello world\n");

    gpu2sched_channel.write(false);
    gpu2sched_channel.write(job);
}

class HelloWorldJob : public llis::Job {
  public:
    size_t get_input_size() override {
        return 5;
    }

    size_t get_output_size() override {
        return 11;
    }

    size_t get_param_size() override {
        return 4;
    }

    void full_init(void* io_ptr) override {
        io_ptr_ = io_ptr;
    }

    void run_next() override {
        ++num_;

        helloworld<<<1, 1>>>(this, gpu2sched_channel_.fork());
    }

    bool has_next() override {
        return num_ < 5;
    }

  private:
    void* io_ptr_;
    int num_ = 0;
};

extern "C" {

llis::Job* init_job() {
    return new HelloWorldJob();
}

}

