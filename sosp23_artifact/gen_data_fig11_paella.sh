#!/bin/bash

res_dir=$1

cd "$(dirname "$0")"/..
abs_path="`pwd`"

cd release/src/server

SERVER_PID=0

trap "kill $SERVER_PID; exit" INT

for ln_sigma in {1.5,2}; do
    for sched in {SS,MS-jbj,MS-kbk,Full}; do
        case $sched in
            SS)
                sched=fifo
                num_streams=1
                suffix=ss
                ;;
            MS-jbj)
                sched=fifo
                num_streams=500
                suffix=msjbj
                ;;
            MS-kbk)
                sched=fifo2
                num_streams=500
                suffix=mskbk
                ;;
            Full)
                sched=full3
                num_streams=500
                suffix=full
                ;;
        esac

        for i in {1000,1053,1111,1176,1250,1333,1429,1538,1667,1818,2000,2222,2500,2857,3333,4000,5000,6667,10000,20000,40000,80000,160000}; do
            taskset -c 4 ./server \
                --name server \
                --sched $sched \
                --num_streams $num_streams &
            SERVER_PID=$!
            sleep 5

            ../../tests/client/test_client_concurrent_run_latencies_fixed_num_multi \
                --server_name server \
                --iat $i \
                --ln_sigma $ln_sigma \
                --start_record_num 0 \
                --seed 1 \
                --prefix "${res_dir}/all_prop${suffix}" \
                --fairness 1000000 \
                --iat_n \
                --iat_g \
                --ln_sigma_n \
                --num_jobs 3000 \
                --concurrency 187 \
                "${abs_path}/release/jobs/tvm_mobilenet/libjob_tvm_mobilenet.so" 0.257 48 \
                "${abs_path}/release/jobs/tvm_densenet121/libjob_tvm_densenet121.so" 0.0706 13 \
                "${abs_path}/release/jobs/tvm_googlenet/libjob_tvm_googlenet.so" 0.0546 10 \
                "${abs_path}/release/jobs/tvm_inception_v3/libjob_tvm_inception_v3.so" 0.0138 3 \
                "${abs_path}/release/jobs/tvm_resnet18/libjob_tvm_resnet18.so" 0.272 51 \
                "${abs_path}/release/jobs/tvm_resnet34/libjob_tvm_resnet34.so" 0.168 31 \
                "${abs_path}/release/jobs/tvm_resnet50/libjob_tvm_resnet50.so" 0.0745 14 \
                "${abs_path}/release/jobs/tvm_squeezenet1_1/libjob_tvm_squeezenet1_1.so" 0.0894999999999999 17
            wait
        done
    done
done
