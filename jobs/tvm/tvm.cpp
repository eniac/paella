#include <llis/job/coroutine_job.h>
#include <llis/job/context.h>

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include <iostream>

class TVMJob : public llis::job::CoroutineJob {
  public:
    size_t get_input_size() override {
        return 28*28 * sizeof(float);
    }

    size_t get_output_size() override {
        return 10 * sizeof(float);
    }

    size_t get_param_size() override {
        return 4;
    }

    void one_time_init() override {
        ctx_gpu_ = DLContext{kDLGPU, 0};
        mod_factory_ = tvm::runtime::Module::LoadFromFile("model-pack.so");
        gmod_ = mod_factory_.GetFunction("default")(ctx_gpu_);
        run_ = gmod_.GetFunction("run");
        tvm::runtime::PackedFunc get_input = gmod_.GetFunction("get_input");
        tvm::runtime::PackedFunc get_output = gmod_.GetFunction("get_output");
        input_dev = get_input("Input3");
        output_dev = get_output(0);
    }

    void body(void* io_ptr) override {
        // TODO: set input, etc
        set_is_mem();
        yield();
        cudaMemcpyAsync(input_dev->data, io_ptr, get_input_size(), cudaMemcpyHostToDevice, get_cuda_stream());
        unset_is_mem();

        run_();

        set_is_mem();
        set_pre_notify();
        yield();
        cudaMemcpyAsync((char*)io_ptr + get_input_size(), input_dev->data, get_output_size(), cudaMemcpyDeviceToHost, get_cuda_stream());
    }

  private:
    DLContext ctx_gpu_;
    tvm::runtime::Module mod_factory_;
    tvm::runtime::Module gmod_;
    tvm::runtime::PackedFunc run_;
    tvm::runtime::NDArray input_dev;
    tvm::runtime::NDArray output_dev;
};

extern "C" {

llis::job::Job* init_job() {
    return new TVMJob();
}

}

