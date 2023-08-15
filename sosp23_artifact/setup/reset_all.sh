#!/bin/bash

PREFIX=/bigdisk

while getopts 'p:' opt; do
  case "$opt" in
    p)
      PREFIX="$OPTARG"
      ;;

    ?|h)
      echo "Usage: $(basename $0) [-p PREFIX]"
      exit 1
      ;;
  esac
done

rm -rf ${PREFIX}/src
rm -rf ${PREFIX}/opt
rm -rf ${PREFIX}/results
rm -rf ${PREFIX}/results-triton
rm -rf ${PREFIX}/results-cuda
rm -rf ${PREFIX}/results-mps
rm -rf ${PREFIX}/graphs
rm -rf ${PREFIX}/models/cuda
rm -rf ${PREFIX}/models/cuda_llis
rm -rf ${PREFIX}/models/tensorflow
rm ~/.bash_profile

