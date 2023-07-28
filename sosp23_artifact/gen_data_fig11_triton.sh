#!/bin/bash

client_path=/bigdisk/src/triton-client-llis
res_dir=/bigdisk/results-triton/

while getopts 'p:o:' opt; do
  case "$opt" in
    p)
      client_path="$OPTARG"
      ;;

    o)
      res_dir="$OPTARG"
      ;;
   
    ?|h)
      echo "Usage: $(basename $0) [-p client_path] [-o output_dir]"
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"

mkdir -p $res_dir/1.5
mkdir -p $res_dir/2

$client_path/run.py -b $client_path/build/cc-clients/examples/grpc_async_infer_client_mixed -o $res_dir/ 10 50 10 $client_path/schedules/newmix3_sops23.yaml

