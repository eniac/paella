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

rm -rf ${PREFIX}/src/llis
rm -rf ${PREFIX}/src/tvm-llis

rm -rf ${PREFIX}/opt/llis
rm -rf ${PREFIX}/opt/tvm-llis

