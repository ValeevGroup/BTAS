#; -*-CMake-*-

cmake_minimum_required (VERSION 2.8.7)
project (BTAS)

# Configure options
option(BTAS_BUILD_UNITTEST "Causes building BTAS unit tests" ON)
option(BTAS_EXPERT "BTAS Expert mode: disables automatically downloading or building dependencies" OFF)
option(BTAS_ASSERT_THROWS "Whether BTAS_ASSERT should throw" OFF)

# Set BTAS version
set(BTAS_MAJOR_VERSION 0)
set(BTAS_MINOR_VERSION 1)
set(BTAS_MICRO_VERSION 0)
set(BTAS_BUILDID alpha)
set(BTAS_VERSION "${BTAS_MAJOR_VERSION}.${BTAS_MINOR_VERSION}.${BTAS_MICRO_VERSION}-${BTAS_BUILDID}")
set(TARGET_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# extra cmake files are shipped with BTAS
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)

include(CMakePushCheckState)
include(GNUInstallDirs)
include(AppendFlags)

##########################
# Standard build variables
##########################

# Get standard build variables from the environment if they have not already been set
if(NOT CMAKE_C_FLAGS OR NOT DEFINED CMAKE_C_FLAGS)
  set(CMAKE_C_FLAGS "$ENV{CPPFLAGS}")
  append_flags(CMAKE_C_FLAGS "$ENV{CFLAGS}")
endif()
if(NOT CMAKE_CXX_FLAGS OR NOT DEFINED CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "$ENV{CPPFLAGS}")
  append_flags(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS}")
endif()
if(NOT CMAKE_EXE_LINKER_FLAGS OR NOT DEFINED CMAKE_EXE_LINKER_FLAGS)
  set(CMAKE_EXE_LINKER_FLAGS "$ENV{LDFLAGS}")
endif()

enable_language (CXX)
if (NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler not found")
endif()

# Set the default fortran integer type.
set(INTEGER4 "FALSE" CACHE BOOL "Set the default Fortran integer type to integer*4")
mark_as_advanced(INTEGER4)

set(CMAKE_SKIP_RPATH FALSE)
set(BUILD_TESTING FALSE CACHE BOOLEAN "BUILD_TESTING")
set(BUILD_TESTING_STATIC FALSE CACHE BOOLEAN "BUILD_TESTING_STATIC")
set(BUILD_TESTING_SHARED FALSE CACHE BOOLEAN "BUILD_TESTING_SHARED")

##########################
# Check C++11 features
##########################
include(TestCXXAcceptsFlag)
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)

# Check for general C++11 support
CHECK_CXX_SOURCE_COMPILES(
    "
    #if (__cplusplus < 201103L) || defined(__INTEL_CXX11_MODE__)
    #error C++11 support not enabled.
    #endif
    int main(){ return 0; }
    "
    BTAS_HAS_CXX11)

# if no C++11, try flags
if (NOT BTAS_HAS_CXX11)
  CHECK_CXX_ACCEPTS_FLAG("-std=c++11" CXX_ACCEPTS_STDCXX11)
  if (CXX_ACCEPTS_STDCXX11)
    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
  else (CXX_ACCEPTS_STDCXX11)
    CHECK_CXX_ACCEPTS_FLAG("-std=c++0x" CXX_ACCEPTS_STDCXX0X)
    if (CXX_ACCEPTS_STDCXX0X)
    set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
    else (CXX_ACCEPTS_STDCXX0X)
      message(FATAL_ERROR "Compiler does not provide C++11 support; provide the appropriate command-line flags explicitly via -DCMAKE_CXX_FLAGS=\"\" or try a newer compiler")
    endif (CXX_ACCEPTS_STDCXX0X)
  endif (CXX_ACCEPTS_STDCXX11)
  
  # look for C++11 support again
  CHECK_CXX_SOURCE_COMPILES(
    "
    #if (__cplusplus < 201103L) || defined(__INTEL_CXX11_MODE__)
    #error C++11 support not enabled.
    #endif
    int main(){ return 0; }
    "
    BTAS_HAS_CXX11_USING_STDFLAG)
  if (NOT BTAS_HAS_CXX11_USING_STDFLAG)
    message(FATAL_ERROR "Compiler cannot compile C++11 code with -std=c++11 or -std=c++0x flags; provide the appropriate command-line flags explicitly via -DCMAKE_CXX_FLAGS=\"\" or try a newer compiler")
  endif (NOT BTAS_HAS_CXX11_USING_STDFLAG)
endif (NOT BTAS_HAS_CXX11)

if(BTAS_HAS_CXX11)
  # static_assert is desired, but an implementation is included
  CHECK_CXX_SOURCE_COMPILES("
    int main(int argc, char* argv[]) {
      static_assert(true, \"don't worry, be happy\");
      return 0;
    }"
    BTAS_HAVE_STATIC_ASSERT)  
  
  # check for variadic templates
  CHECK_CXX_SOURCE_COMPILES("
    #include <iostream>
    #include <iterator>
    #include <algorithm>
    template <typename... args> void print_numbers(const args&... numbers) {
      const int n = sizeof...(args);
      int nums[sizeof...(args)] = { static_cast<int>(numbers)... };
      std::copy(nums, nums+n, std::ostream_iterator<int>(std::cout, \" \"));
    }
    int main(int argc, char* argv[]) {
      print_numbers(-1);
      print_numbers(0, 1, 2, 3, 4);
      return 0;
    }"
    BTAS_HAVE_VARIADIC_TEMPLATES) 
 
endif(BTAS_HAS_CXX11)

# Check type support
include(CheckTypeSize)
check_type_size("long double" BTAS_HAS_LONG_DOUBLE)
check_type_size("long long" BTAS_HAS_LONG_LONG)

##########################
# external dependencies
##########################
add_custom_target(External)

if (BTAS_BUILD_UNITTEST)
  include(external/lapack.cmake)
  include(external/boost.cmake)
endif()

##########################
# configure BTAS_ASSERT
##########################
if (BTAS_ASSERT_THROWS)
  add_definitions(-DBTAS_ASSERT_THROWS)
endif(BTAS_ASSERT_THROWS)

##########################
# sources
##########################

include_directories(${PROJECT_SOURCE_DIR})

##########################
# checking/testing
##########################
enable_testing()
if (BTAS_BUILD_UNITTEST)
  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
  add_subdirectory(unittest)
else(BTAS_BUILD_UNITTEST)
  add_custom_target(check COMMAND echo "WARNING: unit testing disabled. To enable, give -DBTAS_BUILD_UNITTEST=TRUE to cmake __AND__ ensure you have Boost libraries")
endif(BTAS_BUILD_UNITTEST)

# Add target to allow on-the-fly switching of build type

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

