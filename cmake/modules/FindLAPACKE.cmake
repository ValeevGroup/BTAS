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
IF (NOT CBLAS_FOUND)
  FIND_PACKAGE(MKL)
ENDIF (NOT CBLAS_FOUND)
IF (MKL_FOUND AND NOT LAPACKE_LIBRARIES)
  SET(LAPACKE_LIBRARIES ${MKL_LIBRARIES})
  SET(LAPACKE_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  SET(LAPACKE_INCLUDE_FILE "mkl_lapacke.h")
  RETURN ()
ENDIF (MKL_FOUND AND NOT LAPACKE_LIBRARIES)

INCLUDE(CheckLibraryList)

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

# Generic LAPACKE library
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
    MESSAGE(STATUS "LAPACKE library found")
  ELSE(LAPACKE_FOUND)
    MESSAGE(STATUS "LAPACKE library not found.")
  ENDIF(LAPACKE_FOUND)
ENDIF(NOT LAPACKE_FIND_QUIETLY)