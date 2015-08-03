#!/bin/sh

set -e

if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
fi

ls -l /usr/lib
ls -l /usr/lib/libblas

mkdir build_cblas
cd build_cblas
cmake .. -DBTAS_ASSERT_THROWS=ON -DUSE_CBLAS=ON
make VERBOSE=1
make check VERBOSE=1
cd ..

mkdir build
cd build
cmake .. -DBTAS_ASSERT_THROWS=ON
make VERBOSE=1
make check VERBOSE=1
cd ..

