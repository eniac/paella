#include "layer.h"

#include <llis/job/job.h>
#include <llis/job/context.h>
#include <llis/job/utils.h>

class CNNJob : public llis::job::Job {
  public:
    size_t get_input_size() override {
        return 28*28 * sizeof(float);
    }

    size_t get_output_size() override {
        return 10 * sizeof(float);
    }

    size_t get_param_size() override {
        return 0;
    }

    void full_init(void* io_ptr) override {
        input_ = reinterpret_cast<float*>(io_ptr);
        output_ = reinterpret_cast<float*>(reinterpret_cast<char*>(io_ptr) + get_output_offset());

        l_input = Layer(0, 0, 28*28);
        l_c1 = Layer(5*5, 6, 24*24*6);
        l_s1 = Layer(4*4, 1, 6*6*6);
        l_f = Layer(6*6*6, 10, 10);

        llis::job::memset_res(sizeof(float) * l_input.O, this);
    }

    void run_next() override {
        if (stage_ == 0) {
            llis::job::memset(l_input.output, 0x00, sizeof(float) * l_input.O, this, llis::job::Context::get_gpu2sched_channel());
            llis::job::memset_res(sizeof(float) * l_input.O, this);
        } else if (stage_ == 1) {
            llis::job::memset(l_input.preact, 0x00, sizeof(float) * l_input.O, this, llis::job::Context::get_gpu2sched_channel());
            llis::job::memset_res(sizeof(float) * l_c1.O, this);
        } else if (stage_ == 2) {
            llis::job::memset(l_c1.output, 0x00, sizeof(float) * l_c1.O, this, llis::job::Context::get_gpu2sched_channel());
            llis::job::memset_res(sizeof(float) * l_c1.O, this);
        } else if (stage_ == 3) {
            llis::job::memset(l_c1.preact, 0x00, sizeof(float) * l_c1.O, this, llis::job::Context::get_gpu2sched_channel());
            llis::job::memset_res(sizeof(float) * l_s1.O, this);
        } else if (stage_ == 4) {
            llis::job::memset(l_s1.output, 0x00, sizeof(float) * l_s1.O, this, llis::job::Context::get_gpu2sched_channel());
            llis::job::memset_res(sizeof(float) * l_s1.O, this);
        } else if (stage_ == 5) {
            llis::job::memset(l_s1.preact, 0x00, sizeof(float) * l_s1.O, this, llis::job::Context::get_gpu2sched_channel());

            set_num_blocks(64);
            set_num_threads_per_block(64);
            set_smem_size_per_block(0);
            set_num_registers_per_thread(32);
        } else if (stage_ == 6) {
            fp_preact_c1<<<64, 64, 0, get_cuda_stream()>>>((float (*)[28])input_, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight, this, llis::job::Context::get_gpu2sched_channel()->fork());
        } else if (stage_ == 7) {
            fp_bias_c1<<<64, 64, 0, get_cuda_stream()>>>((float (*)[24][24])l_c1.preact, l_c1.bias, this, llis::job::Context::get_gpu2sched_channel()->fork());
        } else if (stage_ == 8) {
            apply_step_function<<<64, 64, 0, get_cuda_stream()>>>(l_c1.preact, l_c1.output, l_c1.O, this, llis::job::Context::get_gpu2sched_channel()->fork());
        } else if (stage_ == 9) {
            fp_preact_s1<<<64, 64, 0, get_cuda_stream()>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight, this, llis::job::Context::get_gpu2sched_channel()->fork());
        } else if (stage_ == 10) {
            fp_bias_s1<<<64, 64, 0, get_cuda_stream()>>>((float (*)[6][6])l_s1.preact, l_s1.bias, this, llis::job::Context::get_gpu2sched_channel()->fork());
        } else if (stage_ == 11) {
            apply_step_function<<<64, 64, 0, get_cuda_stream()>>>(l_s1.preact, l_s1.output, l_s1.O, this, llis::job::Context::get_gpu2sched_channel()->fork());
        } else if (stage_ == 12) {
            fp_preact_f<<<64, 64, 0, get_cuda_stream()>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight, this, llis::job::Context::get_gpu2sched_channel()->fork());
        } else if (stage_ == 13) {
            fp_bias_f<<<64, 64, 0, get_cuda_stream()>>>(l_f.preact, l_f.bias, this, llis::job::Context::get_gpu2sched_channel()->fork());
        } else if (stage_ == 14) {
            apply_step_function<<<64, 64, 0, get_cuda_stream()>>>(l_f.preact, output_, l_f.O, this, llis::job::Context::get_gpu2sched_channel()->fork());
        }

        ++stage_;
    }

    bool has_next() const override {
        return stage_ <= 14;
    }

  private:
    Layer l_input;
    Layer l_c1;
    Layer l_s1;
    Layer l_f;

    int stage_ = 0;
    float* input_;
    float* output_;
};

extern "C" {

llis::job::Job* init_job() {
    return new CNNJob();
}

}

