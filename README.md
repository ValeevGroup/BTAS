Description
===========

Basic Tensor Algebra Subroutines (BTAS) is a C++ headers-only library for tensor algebra. BTAS is a reference implementation of Tensor Working Group concept spec.

Prerequisites
=============

* C++17 compiler
* Boost C++ libraries
  - Iterator
  - (optional) Container for fast small vectors
  - (optional) Serialization for serialization (not header-only)
* (optional) CBLAS+LAPACKe libraries for optimized operations
* (optional) CMake to build and run unit tests

To compile unit tests in the source directory:
* cmake .
* make check
