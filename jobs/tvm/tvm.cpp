#include <llis/job/coroutine_job.h>

#include <tvm/runtime/module.h>

#include <iostream>

class TVMJob : public llis::job::CoroutineJob {
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

    void one_time_init() override {
        ctx_gpu_ = DLContext{kDLGPU, 0};
        mod_factory_ = tvm::runtime::Module::LoadFromFile("model-pack.so");
        gmod_ = mod_factory_.GetFunction("default")(ctx_gpu_);
        run_ = gmod_.GetFunction("run");
    }

    void body(void* io_ptr) override {
        io_ptr_ = io_ptr;

        // TODO: set input, etc
        run_();
    }

  private:
    void* io_ptr_;

    DLContext ctx_gpu_;
    tvm::runtime::Module mod_factory_;
    tvm::runtime::Module gmod_;
    tvm::runtime::PackedFunc run_;
};

extern "C" {

llis::job::Job* init_job() {
    return new TVMJob();
}

}

