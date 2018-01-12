#; -*-CMake-*-

cmake_minimum_required (VERSION 3.1.0)
project (BTAS)

include(FeatureSummary)

# Configure options
option(BTAS_BUILD_UNITTEST "Causes building BTAS unit tests" ON)
option(BTAS_EXPERT "BTAS Expert mode: disables automatically downloading or building dependencies" OFF)
option(BTAS_ASSERT_THROWS "Whether BTAS_ASSERT should throw" OFF)

set(TARGET_MAX_INDEX_RANK 6 CACHE STRING "Determines the rank for which the default BTAS index type will use stack (default: 6); this requires Boost.Container")
add_feature_info("TARGET_MAX_INDEX_RANK=${TARGET_MAX_INDEX_RANK}" TRUE "default BTAS index type will use stack for rank<=${TARGET_MAX_INDEX_RANK}")

# Set BTAS version
set(BTAS_MAJOR_VERSION 0)
set(BTAS_MINOR_VERSION 1)
set(BTAS_MICRO_VERSION 0)
set(BTAS_BUILDID alpha)
set(BTAS_VERSION "${BTAS_MAJOR_VERSION}.${BTAS_MINOR_VERSION}.${BTAS_MICRO_VERSION}-${BTAS_BUILDID}")
set(TARGET_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

# extra cmake files are shipped with BTAS
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

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
# We use C++11 features
##########################
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Check type support
include(CheckTypeSize)
check_type_size("long double" BTAS_HAS_LONG_DOUBLE)
check_type_size("long long" BTAS_HAS_LONG_LONG)

##########################
# external dependencies
##########################
add_custom_target(External)

if (BTAS_BUILD_UNITTEST)
  if (USE_CBLAS)
    find_package(CBLAS REQUIRED)
    find_package(LAPACKE REQUIRED)
    add_definitions(-D_CBLAS_HEADER="${CBLAS_INCLUDE_FILE}" -D_LAPACKE_HEADER="${LAPACKE_INCLUDE_FILE}" -DBTAS_HAS_CBLAS=1)
    message(STATUS "** CBLAS_LIBRARIES = ${CBLAS_LIBRARIES}")
    message(STATUS "** CBLAS_INCLUDE_DIR = ${CBLAS_INCLUDE_DIR}")
    message(STATUS "** CBLAS_INCLUDE_FILE = ${CBLAS_INCLUDE_FILE}")
    message(STATUS "** LAPACKE_LIBRARIES = ${LAPACKE_LIBRARIES}")
    message(STATUS "** LAPACKE_INCLUDE_DIR = ${LAPACKE_INCLUDE_DIR}")
    message(STATUS "** LAPACKE_INCLUDE_FILE = ${LAPACKE_INCLUDE_FILE}")
    include_directories(${CBLAS_INCLUDE_DIR} ${LAPACKE_INCLUDE_DIR})
    if (MKL_FOUND)
      add_definitions(-D_HAS_INTEL_MKL=1)
    endif(MKL_FOUND)
  endif(USE_CBLAS)
  include(external/boost.cmake)
endif()

##########################
# configure BTAS_ASSERT
##########################
if (BTAS_ASSERT_THROWS)
  add_definitions(-DBTAS_ASSERT_THROWS=1)
endif(BTAS_ASSERT_THROWS)

##########################
# sources
##########################

include_directories(${PROJECT_SOURCE_DIR})
add_subdirectory(doc)

##########################
# checking/testing
##########################
enable_testing()
if (BTAS_BUILD_UNITTEST)
  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
  add_subdirectory(unittest)
else(BTAS_BUILD_UNITTEST)
  add_custom_target(check COMMAND echo "WARNING: unit testing disabled. To enable, give -DBTAS_BUILD_UNITTEST=TRUE to cmake")
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

feature_summary(WHAT ALL
                DESCRIPTION "=== BTAS Feature Info ===")
                
