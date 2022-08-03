#; -*-CMake-*-

cmake_minimum_required (VERSION 3.14.0)  # FetchContent_MakeAvailable

###############################################################################
# CMake defaults to address key pain points
###############################################################################

# safety net for dev workflow: accidental install will not affect FindOrFetch*
if (NOT DEFINED CACHE{CMAKE_FIND_NO_INSTALL_PREFIX})
  set(CMAKE_FIND_NO_INSTALL_PREFIX ON CACHE BOOL "Whether find_* commands will search CMAKE_INSTALL_PREFIX and CMAKE_STAGING_PREFIX; see https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_NO_INSTALL_PREFIX.html#variable:CMAKE_FIND_NO_INSTALL_PREFIX")
endif()

###############################################################################
# Bring ValeevGroup cmake toolkit
###############################################################################
include(FetchContent)
if (DEFINED PROJECT_BINARY_DIR)
  set(VG_CMAKE_KIT_PREFIX_DIR PROJECT_BINARY_DIR)
else ()
  set(VG_CMAKE_KIT_PREFIX_DIR CMAKE_CURRENT_BINARY_DIR)
endif()
FetchContent_Declare(
        vg_cmake_kit
        QUIET
        GIT_REPOSITORY      https://github.com/ValeevGroup/kit-cmake.git
        SOURCE_DIR ${${VG_CMAKE_KIT_PREFIX_DIR}}/cmake/vg
        BINARY_DIR ${${VG_CMAKE_KIT_PREFIX_DIR}}/cmake/vg-build
        SUBBUILD_DIR ${${VG_CMAKE_KIT_PREFIX_DIR}}/cmake/vg-subbuild
)
FetchContent_MakeAvailable(vg_cmake_kit)
list(APPEND CMAKE_MODULE_PATH "${vg_cmake_kit_SOURCE_DIR}/modules")

###############################################################################
# Announce ourselves
###############################################################################
# see https://semver.org/
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

project(BTAS
        VERSION ${BTAS_VERSION}
        DESCRIPTION "BTAS: Basic Tensor Algebra for Seeplusplus"
        LANGUAGES CXX
        HOMEPAGE_URL "https://github.com/ValeevGroup/BTAS")
enable_language(C)  # C needed even for basic platform introspection

# extra cmake files are shipped with BTAS
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

include(FeatureSummary)
include(RedefaultableOption)
include(CMakePackageConfigHelpers)
include(AddCustomTargetSubproject)
include(CTest) # defines BUILD_TESTING option

# Configure options
redefaultable_option(BTAS_BUILD_DEPS_FROM_SOURCE "Whether to build missing dependencies from source" OFF)
add_feature_info(BUILD_DEPS_FROM_SOURCE BTAS_BUILD_DEPS_FROM_SOURCE "Will build missing dependencies from source")
redefaultable_option(BTAS_ASSERT_THROWS "Whether BTAS_ASSERT should throw; enable if BUILD_TESTING=ON" ${BUILD_TESTING})
add_feature_info(ASSERT_THROWS BTAS_ASSERT_THROWS "BTAS_ASSERT(x) will throw if x is false, and not be affected by NDEBUG")
redefaultable_option(BTAS_USE_BLAS_LAPACK "Whether to enable BLAS/LAPACK bindings via BLAS++/LAPACK++" ON)
add_feature_info(USE_BLAS_LAPACK BTAS_USE_BLAS_LAPACK "Will use BLAS and LAPACK linear algebra distributions via their BLAS++/LAPACK++ interfaces" )
option(ENABLE_WFN91_LINALG_DISCOVERY_KIT "Use linear algebra discovery kit from github.com/wavefunction91 [recommended]" ON)
add_feature_info(WFN91LinearAlgebraDiscoveryKit ENABLE_WFN91_LINALG_DISCOVERY_KIT "Linear algebra discovery kit from github.com/wavefunction91 supports many more corner cases than the default CMake modules and/or ICL's BLAS++/LAPACK++ modules")

set(TARGET_MAX_INDEX_RANK 6 CACHE STRING "Determines the rank for which the default BTAS index type will use stack (default: 6); this requires Boost.Container")
add_feature_info("TARGET_MAX_INDEX_RANK=${TARGET_MAX_INDEX_RANK}" TRUE "default BTAS index type will use stack for rank<=${TARGET_MAX_INDEX_RANK}")

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
# We use C++17 features
##########################
# but insist on strict standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ ISO Standard version")
if (NOT(CMAKE_CXX_STANDARD EQUAL 17 OR CMAKE_CXX_STANDARD EQUAL 20))
  message(FATAL_ERROR "C++ 2017 ISO Standard or higher is required to compile BTAS")
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
target_compile_features(BTAS INTERFACE "cxx_std_17")
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
if( BTAS_USE_BLAS_LAPACK )
  include(external/linalgpp.cmake)
endif()
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
if (BUILD_TESTING)
  add_custom_target_subproject(btas check USES_TERMINAL COMMAND ${CMAKE_CTEST_COMMAND} -V)
  add_subdirectory(unittest)
else(BUILD_TESTING)
  add_custom_target_subproject(btas check COMMAND echo "WARNING: unit testing disabled. To enable, give -DBUILD_TESTING=ON to cmake")
endif(BUILD_TESTING)

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
    COMMAND ${CMAKE_COMMAND} -DCOMPONENT=btas-config -P ${PROJECT_BINARY_DIR}/cmake_install.cmake
    COMMENT "Installing BTAS config components")

feature_summary(WHAT ALL
    DESCRIPTION "=== BTAS Package/Feature Info ===")

###############################################################################
# appendix: misc details
###############################################################################
SET(CMAKE_COLOR_MAKEFILE ON)
