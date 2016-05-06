#!/bin/sh

set -ev

if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
else
    export CC=/usr/bin/clang-3.7
    export CXX=/usr/bin/clang++-3.7
fi

##########   test with cblas   ##########
mkdir build_cblas
cd build_cblas
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

