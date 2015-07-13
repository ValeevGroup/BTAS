#!/bin/sh

set -e

if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
fi

mkdir build
cd build
cmake .. -DBTAS_ASSERT_THROWS=ON
make VERBOSE=1
make check VERBOSE=1
cd ..

mkdir build_cblas
cd build_cblas
ls -l /usr/include/cblas*
cmake .. -DBTAS_ASSERT_THROWS=ON -DUSE_CBLAS=ON
make VERBOSE=1
make check VERBOSE=1
cd ..
