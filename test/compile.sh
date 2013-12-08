#!/bin/bash

SRC=$1
EXE=${SRC%.*}.x

g++ -std=c++11 -static -D_HAS_CBLAS -D_HAS_INTEL_MKL -I.. -I/opt/intel/mkl/include $SRC -o $EXE -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

#
