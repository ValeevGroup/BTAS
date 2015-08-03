#!/bin/sh

set -e

if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
fi

ls -l /usr/lib
ls -l /usr/lib/libblas

##########   test with cblas   ##########
mkdir build_cblas
cd build_cblas

cat <<EOF >test.cc
#include<cblas.h>

int main() {
  double x = cblas_ddot(32, (double*)nullptr, 1, (double*)nullptr, 1);
  return 0;
}
EOF

$CXX -std=c++11 test.cc -lblas -o test

cmake .. -DBTAS_ASSERT_THROWS=ON -DUSE_CBLAS=ON
make VERBOSE=1
make check VERBOSE=1
cd ..

########## test without cblas ##########
mkdir build
cd build
cmake .. -DBTAS_ASSERT_THROWS=ON
make VERBOSE=1
make check VERBOSE=1
cd ..

