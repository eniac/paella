#include <llis/job/coroutine_job.h>

#include <tvm/runtime/module.h>

#include <iostream>

class TVMJob : public llis::job::CoroutineJob {
  public:
    TVMJob() {
        set_smem_size_per_block(0);
    }

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

        auto start_time = std::chrono::steady_clock::now();

        ctx_gpu_ = DLContext{kDLGPU, 0};
        mod_factory_ = tvm::runtime::Module::LoadFromFile("model-pack.so");
        gmod_ = mod_factory_.GetFunction("default")(ctx_gpu_);
        run_ = gmod_.GetFunction("run");

        auto end_time = std::chrono::steady_clock::now();

        std::cout << "Time taken for TVM startup: " << std::chrono::duration<double, std::micro>(end_time - start_time).count() << std::endl;

        CoroutineJob::full_init(io_ptr);
    }

    void body() override {
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

