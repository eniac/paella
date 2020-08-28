#include <llis/job.h>

#include <memory>

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
};

extern "C" {

llis::Job* init_job() {
    return new HelloWorldJob();
}

}

