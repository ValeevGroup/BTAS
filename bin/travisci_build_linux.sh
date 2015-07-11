#!/bin/sh

set -e

if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
fi

mkdir build_nolapack
cd build_nolapack
cmake .. -DBTAS_ASSERT_THROWS=ON
make VERBOSE=1
make check VERBOSE=1
cd ..

mkdir build_lapack
cd build_lapack
cmake .. -DBTAS_ASSERT_THROWS=ON -DUSE_LAPACK=ON
make VERBOSE=1
make check VERBOSE=1
cd ..
