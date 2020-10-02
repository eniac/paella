#include <llis/job.h>

#include <cstdio>

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

    bool run_next() override {
        printf("hello world\n");
        return false;
    }

  private:
    void* io_ptr_;

};

extern "C" {

llis::Job* init_job() {
    return new HelloWorldJob();
}

}

