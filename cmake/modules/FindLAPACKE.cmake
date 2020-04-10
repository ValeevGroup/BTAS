# based on: https://github.com/clementfarabet/THC/blob/master/scripts/FindCBLAS.cmake

# - Find LAPACKE interface to LAPACK library
#
# This module finds an installed library that implements the LAPACKE interface
# (see http://www.netlib.org/lapack/lapacke.html) to the LAPACK library.
#
# This module sets the following variables:
#  LAPACKE_FOUND - set to true if a library implementing the LAPACKE interface is found
#  LAPACKE_LIBRARIES - list of libraries (using full path name) to link against to use LAPACKE
#  LAPACKE_INCLUDE_DIR - path to includes
#  LAPACKE_INCLUDE_FILE - the file to be included to use LAPACKE
#

SET(LAPACKE_LIBRARIES)
SET(LAPACKE_INCLUDE_DIR)
SET(LAPACKE_INCLUDE_FILE)

# Unless already found look for LAPACKE in Intel mkl
IF (BTAS_ENABLE_MKL)
  IF (NOT CBLAS_FOUND)
    FIND_PACKAGE(MKL)
  ENDIF (NOT CBLAS_FOUND)
  IF (MKL_FOUND AND NOT LAPACKE_LIBRARIES)
    SET(LAPACKE_LIBRARIES ${MKL_LIBRARIES})
    SET(LAPACKE_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    SET(LAPACKE_INCLUDE_FILE "mkl_lapacke.h")
    RETURN ()
  ENDIF (MKL_FOUND AND NOT LAPACKE_LIBRARIES)
ENDIF(BTAS_ENABLE_MKL)

INCLUDE(CheckLibraryList)

# initialize BLA_STATIC, if needed, and adjust the library suffixes search list
if (BUILD_SHARED_LIBS)
  set(_bla_static FALSE)
else (BUILD_SHARED_LIBS)
  set(_bla_static TRUE)
endif (BUILD_SHARED_LIBS)
set(BLA_STATIC ${_bla_static} CACHE BOOL "Whether to use static linkage for BLAS, LAPACK, and related libraries")
if (BLA_STATIC)
  list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 "${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif(BLA_STATIC)

# Apple does not provide LAPACKE as part of Accelerate/vecLib

# LAPACKE in ATLAS library? (http://math-atlas.sourceforge.net/)
IF(NOT LAPACKE_LIBRARIES)
  CHECK_LIBRARY_LIST(
    LAPACKE_LIBRARIES
    LAPACKE
    LAPACKE_dgesv
    ""
    "lapacke;cblas;atlas"
    "lapacke.h"
    TRUE )
ENDIF(NOT LAPACKE_LIBRARIES)

# NETLIB LAPACKE library (w/o Fortran)
IF(NOT LAPACKE_LIBRARIES)
  CHECK_LIBRARY_LIST(
      LAPACKE_LIBRARIES
      LAPACKE
      LAPACKE_dgesv
      ""
      "lapacke;lapack;blas"
      "lapacke.h"
      TRUE )
ENDIF()

# NETLIB LAPACKE library (w/ Fortran)
get_property(_project_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
IF((NOT LAPACKE_LIBRARIES) AND ("Fortran" IN_LIST _project_languages OR CMAKE_Fortran_COMPILER))
  if (NOT("Fortran" IN_LIST _project_languages))
    enable_language(Fortran)
  endif()
  set(_current_CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH}")
  list(APPEND CMAKE_LIBRARY_PATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
  CHECK_LIBRARY_LIST(
      LAPACKE_LIBRARIES
      LAPACKE
      LAPACKE_dgesv
      ""
      "lapacke;lapack;blas;libgfortran${CMAKE_SHARED_LIBRARY_SUFFIX};libm${CMAKE_SHARED_LIBRARY_SUFFIX};libpthread${CMAKE_SHARED_LIBRARY_SUFFIX}"
      "lapacke.h"
      TRUE )
  set(CMAKE_LIBRARY_PATH "${_current_CMAKE_LIBRARY_PATH}")
ENDIF()

# Generic LAPACKE library (w/o any prereqs)
IF(NOT LAPACKE_LIBRARIES)
  CHECK_LIBRARY_LIST(
    LAPACKE_LIBRARIES
    LAPACKE
    LAPACKE_dgesv
    ""
    "lapacke"
    "lapacke.h"
    TRUE )
ENDIF(NOT LAPACKE_LIBRARIES)

IF(LAPACKE_LIBRARIES)
  SET(LAPACKE_FOUND TRUE)
ELSE(LAPACKE_LIBRARIES)
  SET(LAPACKE_FOUND FALSE)
ENDIF(LAPACKE_LIBRARIES)

IF(NOT LAPACKE_FOUND AND LAPACKE_FIND_REQUIRED)
  MESSAGE(FATAL_ERROR "LAPACKE API not found. Please specify library location")
ENDIF(NOT LAPACKE_FOUND AND LAPACKE_FIND_REQUIRED)
IF(NOT LAPACKE_FIND_QUIETLY)
  IF(LAPACKE_FOUND)
    MESSAGE(STATUS "LAPACKE library found: LAPACKE_LIBRARIES=${LAPACKE_LIBRARIES}")
  ELSE(LAPACKE_FOUND)
    MESSAGE(STATUS "LAPACKE library not found.")
  ENDIF(LAPACKE_FOUND)
ENDIF(NOT LAPACKE_FIND_QUIETLY)

if (BLA_STATIC)
  list(REMOVE_AT CMAKE_FIND_LIBRARY_SUFFIXES 0)
endif(BLA_STATIC)
