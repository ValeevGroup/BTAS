Description
===========

Basic Tensor Algebra Subroutines (BTAS) is a C++ library for tensor algebra. BTAS is a reference implementation of Tensor Working Group concept spec. The library can be optionally used header-only at the cost of much lower performance for most operations.

Prerequisites
=============

* C++17 compiler
* CMake
* Boost C++ libraries
  - (required) Container, Iterator, Random
  - (optional) Serialization for serialization (non-header-only)
* (used by default, strongly recommended, but can be disabled) BLAS+LAPACK libraries and their BLAS++/LAPACK++ C++ APIs for optimized operations (non-header-only)

Building and Installing
=======================
TL;DR version
* `cmake -S /path/to/BTAS/srcdir -B /path/to/BTAS/builddir -DCMAKE_PREFIX_PATH="/path/to/boost;/path/to/blas/and/lapack"`
* optional: `cmake --build /path/to/BTAS/builddir --target check`
* if configured with `-DBTAS_BUILD_DEPS_FROM_SOURCE=ON`: `cmake --build /path/to/BTAS/builddir --target build-boost-in-BTAS`
* `cmake --build /path/to/BTAS/builddir --target install`

## obtaining prerequisites
* Linear algebra (BLAS+LAPACK): should come with your dev toolchain (e.g., on MacOS) or can be installed using the system package manager or as a vendor-provided package (e.g., Intel Math Kernel Libraries)
* Boost:
  - It is recommended to use a package manager to install Boost. This can be doneas follows:
    - APT package manager (e.g., on Ubuntu Linux): `apt-get install libboost-all-dev`
    - Homebrew package manager (on MacOS) via `brew install boost`.
  - You can also try to build Boost yourself by following instructions [here](https://www.boost.org/doc/libs/1_84_0/more/getting_started/unix-variants.html).
  - Last resort is to let BTAS build Boost from source, as a CMake _subproject_ using [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html). Unfortunately, Boost's [emerging CMake harness](https://github.com/boostorg/cmake/) used to build it is not yet fully functional, hence may not be as robust as desired. Here are some ints:
    - Configure with CMake cache variable `BTAS_BUILD_DEPS_FROM_SOURCE` to `ON` (either via command line or the `ccmake` GUI) 
    - If BTAS is the top-level CMake project (i.e. it is not being built as a subproject itself) installing BTAS by building its install target may not build Boost libraries automatically. Thus the user may need to build `build-boost-in-BTAS` target manually before building `install` target.

## useful CMake variables
- `CMAKE_CXX_COMPILER` -- specifies the C++ compiler (by default CMake will look for the C++ compiler in `PATH`)
- `CMAKE_INSTALL_PREFIX` -- specifies the installation prefix (by default CMake will install to `/usr/local`)
- `CMAKE_BUILD_TYPE` -- specifies the build type (by default CMake will build in `Release` mode)
- `CMAKE_PREFIX_PATH` -- semicolon-separated list of paths specifying the locations of dependencies
- `BTAS_USE_BLAS_LAPACK` -- specifies whether to enable the use of BLAS/LAPACK via the BLAS++/LAPACK++ APIs; the default is `ON`
- `BTAS_BUILD_DEPS_FROM_SOURCE` -- specifies whether to enable building the missing dependencies (Boost) from source; the default is `OFF`
- `BUILD_TESTING` -- specifies whether to build unit tests; the default is `ON`
- `TARGET_MAX_INDEX_RANK` -- specifies the rank for which the default BTAS index type will use stack; the default is `6`
