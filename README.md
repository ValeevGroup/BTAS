Description
===========

Basic Tensor Algebra Subroutines (BTAS) is a C++ library for tensor algebra. BTAS is a reference implementation of Tensor Working Group concept spec. The library can be optionally used header-only at the cost of much lower performance for most operations.

Prerequisites
=============

* C++17 compiler
* CMake
* Boost C++ libraries
  - Iterator
  - (optional, but recommended) Container for fast small vectors
  - (optional) Serialization for serialization (non-header-only)
* (used by default, strongly recommended, but can be disabled) BLAS+LAPACK libraries and their BLAS++/LAPACK++ C++ APIs for optimized operations (non-header-only)

Building and Installing
=======================
TL;DR version
* cmake .
* make check

## useful CMake variables
- `CMAKE_CXX_COMPILER` -- specifies the C++ compiler (by default CMake will look for the C++ compiler in `PATH`)
- `BTAS_USE_BLAS_LAPACK` -- specifies whether to enable the use of BLAS/LAPACK via the BLAS++/LAPACK++ APIs; the default is `ON`
- `BTAS_BUILD_DEPS_FROM_SOURCE` -- specifies whether to enable building the missing dependencies (Boost) from source; the default is `OFF`
- `BUILD_TESTING` -- specifies whether to build unit tests; the default is `ON`
- `TARGET_MAX_INDEX_RANK` -- specifies the rank for which the default BTAS index type will use stack; the default is `6`
