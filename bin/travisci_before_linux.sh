#!/bin/sh

set -ev

# Add PPA for a newer version GCC
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
# Add PPA for newer cmake (3.2.3)
sudo add-apt-repository ppa:george-edison55/precise-backports -y
# Add repo with newer clang
sudo add-apt-repository 'deb http://llvm.org/apt/precise/ llvm-toolchain-precise-3.7 main'
sudo wget -O - http://llvm.org/apt/llvm-snapshot.gpg.key|sudo apt-key add -
# Add PPA for a newer boost
sudo add-apt-repository ppa:boost-latest/ppa -y

# Update package list
sudo apt-get update -qq

# Install CMake 3
sudo apt-get -qq -y --no-install-suggests --no-install-recommends --force-yes install cmake cmake-data
cmake --version

sudo apt-get install -qq -y g++-$GCC_VERSION clang-3.7 libboost1.55-dev libboost-serialization1.55-dev libblas-dev liblapack-dev
