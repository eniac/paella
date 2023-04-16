#!/bin/bash

cd "$(dirname "$0")"
script_path=`pwd`
cd -

python3 $script_path/plot_latency_throughput_subplots.py \
    -o all_prop_newmix3_no_mnist_lns2.pdf \
    -i all_prop_direct_multistream_newmix3_no_mnist_1_lns2.txt \
    -a 'CUDA-MS' \
    -i all_prop_direct_singlestream_newmix3_no_mnist_1_lns2.txt \
    -a 'CUDA-SS' \
    -i all_prop_singlestream_newmix3_no_mnist_fifo_1_lns2.txt \
    -a 'Paella-SS' \
    -i all_prop_newmix3_no_mnist_fifo_2_lns2.txt \
    -a 'Paella-CS-jbj' \
    -i all_prop_newmix3_no_mnist_fifo2_2_lns2.txt \
    -a 'Paella-CS-kbk' \
    -i all_prop_newmix3_no_mnist_full3_2_lns2.txt \
    -a 'Paella-SRPT' \
    -m 0 -n All \
    -m 1 -n Mobilenet \
    -m 2 -n Densenet \
    -m 3 -n GoogleNet \
    -m 4 -n InceptionV3 \
    -m 5 -n Resnet18 \
    -m 6 -n Resnet34 \
    -m 7 -n Resnet50 \
    -m 8 -n SqueezeNet \
    --xaxis throughput \
    --yaxis p99 \
    --subplotx 3 \
    --subploty 3 \
    --ylim 2000

python3 $script_path/plot_latency_throughput_subplots.py \
    -o all_prop_newmix3_no_mnist_lns1.5.pdf \
    -i all_prop_direct_multistream_newmix3_no_mnist_1_lns1.5.txt \
    -a 'CUDA-MS' \
    -i all_prop_direct_singlestream_newmix3_no_mnist_1_lns1.5.txt \
    -a 'CUDA-SS' \
    -i all_prop_singlestream_newmix3_no_mnist_fifo_1_lns1.5.txt \
    -a 'Paella-SS' \
    -i all_prop_newmix3_no_mnist_fifo_2_lns1.5.txt \
    -a 'Paella-CS-jbj' \
    -i all_prop_newmix3_no_mnist_fifo2_2_lns1.5.txt \
    -a 'Paella-CS-kbk' \
    -i all_prop_newmix3_no_mnist_full3_2_lns1.5.txt \
    -a 'Paella-SRPT' \
    -m 0 -n All \
    -m 1 -n Mobilenet \
    -m 2 -n Densenet \
    -m 3 -n GoogleNet \
    -m 4 -n InceptionV3 \
    -m 5 -n Resnet18 \
    -m 6 -n Resnet34 \
    -m 7 -n Resnet50 \
    -m 8 -n SqueezeNet \
    --xaxis throughput \
    --yaxis p99 \
    --subplotx 3 \
    --subploty 3 \
    --ylim 2000

