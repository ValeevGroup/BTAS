#; -*-CMake-*-

cmake_minimum_required (VERSION 3.8.0)  # cxx_std_11 compile feature
project (BTAS)
enable_language(CXX)

# extra cmake files are shipped with BTAS
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

include(FeatureSummary)
include(RedefaultableOption)
include(CMakePackageConfigHelpers)
include(AddCustomTargetSubproject)

# Configure options
redefaultable_option(BTAS_BUILD_UNITTEST "Whether to build unit tests" OFF)
add_feature_info(BUILD_UNITTEST BTAS_BUILD_UNITTEST "Will build unit tests")
redefaultable_option(BTAS_ASSERT_THROWS "Whether BTAS_ASSERT should throw; enable if BTAS_BUILD_UNITTEST=ON" OFF)
add_feature_info(ASSERT_THROWS BTAS_ASSERT_THROWS "BTAS_ASSERT(x) will throw if x is false, and not be affected by NDEBUG")
redefaultable_option(BTAS_USE_CBLAS_LAPACKE "Whether to use BLAS+LAPACK via CBLAS+LAPACKE" ON)
add_feature_info(USE_CBLAS_LAPACKE BTAS_USE_CBLAS_LAPACKE "Will use BLAS and LAPACK linear algebra libraries via their CBLAS and LAPACKE interfaces")
redefaultable_option(BTAS_ENABLE_MKL "Whether to look for Intel MKL" ON)
add_feature_info(ENABLE_MKL BTAS_ENABLE_MKL "Will look for Intel MKL library")

set(TARGET_MAX_INDEX_RANK 6 CACHE STRING "Determines the rank for which the default BTAS index type will use stack (default: 6); this requires Boost.Container")
add_feature_info("TARGET_MAX_INDEX_RANK=${TARGET_MAX_INDEX_RANK}" TRUE "default BTAS index type will use stack for rank<=${TARGET_MAX_INDEX_RANK}")

# Set BTAS version
set(BTAS_MAJOR_VERSION 1)
set(BTAS_MINOR_VERSION 0)
set(BTAS_MICRO_VERSION 0)
set(BTAS_PRERELEASE_ID alpha.1)
set(BTAS_VERSION "${BTAS_MAJOR_VERSION}.${BTAS_MINOR_VERSION}.${BTAS_MICRO_VERSION}")
if (BTAS_PRERELEASE_ID)
  set(BTAS_EXT_VERSION "${BTAS_VERSION}-${BTAS_PRERELEASE_ID}")
else(BTAS_PRERELEASE_ID)
  set(BTAS_EXT_VERSION "${BTAS_VERSION}")
endif(BTAS_PRERELEASE_ID)
set(TARGET_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

include(CMakePushCheckState)
include(GNUInstallDirs)
include(AppendFlags)

##########################
# Standard build variables
##########################
set(BTAS_INSTALL_BINDIR "bin"
    CACHE PATH "BTAS BIN install directory")
set(BTAS_INSTALL_INCLUDEDIR "include"
    CACHE PATH "BTAS INCLUDE install directory")
set(BTAS_INSTALL_LIBDIR "lib"
    CACHE PATH "BTAS LIB install directory")
set(BTAS_INSTALL_SHAREDIR "share/BTAS/${BTAS_MAJOR_VERSION}.${BTAS_MINOR_VERSION}.${BTAS_MICRO_VERSION}"
    CACHE PATH "BTAS SHARE install directory")
set(BTAS_INSTALL_DATADIR "${BTAS_INSTALL_SHAREDIR}/data"
    CACHE PATH "BTAS DATA install directory")
set(BTAS_INSTALL_DOCDIR "${BTAS_INSTALL_SHAREDIR}/doc"
    CACHE PATH "BTAS DOC install directory")
set(BTAS_INSTALL_CMAKEDIR "lib/cmake/BTAS"
    CACHE PATH "BTAS CMAKE install directory")

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
if (NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler not found")
endif()

set(CMAKE_SKIP_RPATH FALSE)

##########################
# We use C++11 features
##########################
# but insist on strict standard
set(CMAKE_CXX_STANDARD 11 CACHE STRING "C++ ISO Standard version")
if (NOT(CMAKE_CXX_STANDARD EQUAL 11 OR CMAKE_CXX_STANDARD EQUAL 14 OR CMAKE_CXX_STANDARD EQUAL 17 OR CMAKE_CXX_STANDARD EQUAL 20))
  message(FATAL_ERROR "C++ 2011 ISO Standard or higher is required to compile BTAS")
endif()
# C++20 is only configurable via compile features with cmake 3.12 and older
if (CMAKE_CXX_STANDARD EQUAL 20 AND CMAKE_VERSION VERSION_LESS 3.12.0)
  cmake_minimum_required (VERSION 3.12.0)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL  "Whether to use extensions of C++ ISO Standard version")

# Check type support
include(CheckTypeSize)
check_type_size("long double" BTAS_HAS_LONG_DOUBLE)
check_type_size("long long" BTAS_HAS_LONG_LONG)

#######################################
# create exportable BTAS library target
#######################################
add_library(BTAS INTERFACE)
target_compile_features(BTAS INTERFACE "cxx_std_11")
target_include_directories(BTAS INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${BTAS_INSTALL_INCLUDEDIR}>)
install(TARGETS BTAS EXPORT btas COMPONENT BTAS)
install(DIRECTORY btas
    COMPONENT BTAS
    DESTINATION "${BTAS_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
                   PATTERN "*.h.in" EXCLUDE
    )

##########################
# external dependencies
##########################
add_custom_target(External-btas)

if (BTAS_USE_CBLAS_LAPACKE)
  find_package(CBLAS REQUIRED)
  find_package(LAPACKE REQUIRED)
  target_compile_definitions(BTAS INTERFACE -DBTAS_CBLAS_HEADER="${CBLAS_INCLUDE_FILE}" -DBTAS_LAPACKE_HEADER="${LAPACKE_INCLUDE_FILE}" -DBTAS_HAS_CBLAS=1 -DBTAS_HAS_LAPACKE=1)
  message(STATUS "** CBLAS_LIBRARIES = ${CBLAS_LIBRARIES}")
  message(STATUS "** CBLAS_INCLUDE_DIR = ${CBLAS_INCLUDE_DIR}")
  message(STATUS "** CBLAS_INCLUDE_FILE = ${CBLAS_INCLUDE_FILE}")
  message(STATUS "** LAPACKE_LIBRARIES = ${LAPACKE_LIBRARIES}")
  message(STATUS "** LAPACKE_INCLUDE_DIR = ${LAPACKE_INCLUDE_DIR}")
  message(STATUS "** LAPACKE_INCLUDE_FILE = ${LAPACKE_INCLUDE_FILE}")
  target_include_directories(BTAS INTERFACE ${CBLAS_INCLUDE_DIR} ${LAPACKE_INCLUDE_DIR})
  if (MKL_FOUND)
    target_compile_definitions(BTAS INTERFACE -DBTAS_HAS_INTEL_MKL=1)
    if (MKL_ILP64)
      target_compile_definitions(BTAS INTERFACE -DMKL_ILP64=1)
    endif(MKL_ILP64)
  endif(MKL_FOUND)
  target_link_libraries(BTAS INTERFACE ${LAPACKE_LIBRARIES} ${CBLAS_LIBRARIES})
endif(BTAS_USE_CBLAS_LAPACKE)

include(external/boost.cmake)

##########################
# configure BTAS_ASSERT
##########################
if (BTAS_ASSERT_THROWS)
  target_compile_definitions(BTAS INTERFACE -DBTAS_ASSERT_THROWS=1)
endif(BTAS_ASSERT_THROWS)

##########################
# dox
##########################
add_subdirectory(doc)

##########################
# checking/testing
##########################
enable_testing(true)
if (BTAS_BUILD_UNITTEST)
  add_custom_target_subproject(btas check USES_TERMINAL COMMAND ${CMAKE_CTEST_COMMAND} -V)
  add_subdirectory(unittest)
else(BTAS_BUILD_UNITTEST)
  add_custom_target_subproject(btas check COMMAND echo "WARNING: unit testing disabled. To enable, give -DBTAS_BUILD_UNITTEST=TRUE to cmake")
endif(BTAS_BUILD_UNITTEST)

##########################
# Get the git revision tag information
##########################

if(EXISTS ${PROJECT_SOURCE_DIR}/.git)
  find_package(Git REQUIRED)
  execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE BTAS_REVISION )
  string(REGEX MATCH "[0-9a-f]*"
      BTAS_REVISION "${BTAS_REVISION}")
else()
  set(BTAS_REVISION "unknown")
endif()

configure_file(
    ${PROJECT_SOURCE_DIR}/btas/version.h.in
    ${PROJECT_BINARY_DIR}/btas/version.h
)
install(FILES ${PROJECT_BINARY_DIR}/btas/version.h
    DESTINATION "${BTAS_INSTALL_INCLUDEDIR}/btas")

# Create the version file
write_basic_package_version_file(btas-config-version.cmake
    VERSION ${BTAS_VERSION} COMPATIBILITY AnyNewerVersion)

# Create the targets file
export(EXPORT btas
       NAMESPACE BTAS::
       FILE "${PROJECT_BINARY_DIR}/btas-targets.cmake")

## Create the configure file
configure_package_config_file(cmake/btas-config.cmake.in
    "${PROJECT_BINARY_DIR}/btas-config.cmake"
    INSTALL_DESTINATION "${BTAS_INSTALL_CMAKEDIR}"
    PATH_VARS CMAKE_INSTALL_PREFIX BTAS_INSTALL_BINDIR
    BTAS_INSTALL_INCLUDEDIR BTAS_INSTALL_LIBDIR
    BTAS_INSTALL_DOCDIR BTAS_INSTALL_CMAKEDIR)

## Install config, version, and target files
install(EXPORT btas
    FILE "btas-targets.cmake"
    DESTINATION "${BTAS_INSTALL_CMAKEDIR}"
    NAMESPACE BTAS::
    COMPONENT btas-config)
install(FILES
    "${PROJECT_BINARY_DIR}/btas-config.cmake"
    "${PROJECT_BINARY_DIR}/btas-config-version.cmake"
    DESTINATION "${BTAS_INSTALL_CMAKEDIR}"
    COMPONENT btas-config)
add_custom_target_subproject(btas install-config
    COMMAND ${CMAKE_COMMAND} -DCOMPONENT=btas-config -P ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMENT "Installing BTAS config components")

feature_summary(WHAT ALL
    DESCRIPTION "=== BTAS Package/Feature Info ===")

###############################################################################
# appendix: misc details
###############################################################################
SET(CMAKE_COLOR_MAKEFILE ON)
