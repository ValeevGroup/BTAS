#!/bin/sh

set -ev

# Environment variables
export CXXFLAGS="-mno-avx"
if [ "$CXX" = "g++" ]; then
    export CC=/usr/bin/gcc-$GCC_VERSION
    export CXX=/usr/bin/g++-$GCC_VERSION
else
    export CC=/usr/bin/clang-$CLANG_VERSION
    export CXX=/usr/bin/clang++-$CLANG_VERSION
fi

cd ${BUILD_PREFIX}

##########   test with blas+lapack   ##########
#mkdir build_cblas
#cd build_cblas
## control whether to use ILP64 or not by the parity of
## GCC_VERSION + CLANG_VERSION
#gccv=$GCC_VERSION
#clangv=$([ "X$CLANG_VERSION" = "X" ] && echo "0" || echo "$CLANG_VERSION")
#ilp64v=$(($gccv+$clangv))
#export PREFER_ILP64=$((ilp64v % 2))
## cannot link against static blas libs reliably so use shared ... other codes may still need to be able to configure BTAS for static linking
#cmake ${TRAVIS_BUILD_DIR} -DBLA_STATIC=OFF -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBTAS_ASSERT_THROWS=ON -DBTAS_BUILD_UNITTEST=ON -DMKL_PREFER_ILP64=${PREFER_ILP64}
#make VERBOSE=1
#make check VERBOSE=1
#cd ..


########## build blaspp + lapackpp    ##########
icl_install_prefix=$PWD/icl_install
git clone https://dbwy@bitbucket.org/icl/blaspp.git
git clone https://dbwy@bitbucket.org/icl/lapackpp.git
cmake -Hblaspp -Bbuild_blaspp -DCMAKE_INSTALL_PREFIX=${icl_install} -Dbuild_tests=OFF
make -C build_blaspp -j2 install
cmake -Hlapackpp -Bbuild_lapackpp -DCMAKE_INSTALL_PREFIX=${icl_install} -DBUILD_LAPACKPP_TESTS=OFF
make -C build_lapackpp -j2 install


########## test without blas+lapack   ##########
mkdir build
cd build
cmake ${TRAVIS_BUILD_DIR} -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_PREFIX_PATH=${icl_install_prefix} -DBTAS_ASSERT_THROWS=ON  -DBTAS_BUILD_UNITTEST=ON
make VERBOSE=1
make check VERBOSE=1
cd ..

