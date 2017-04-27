#!/bin/sh

set -ev

# Print compiler information
$CC --version
$CXX --version

# log the CMake version (need 3+)
cmake --version
