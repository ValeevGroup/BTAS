# based on: https://github.com/clementfarabet/THC/blob/master/scripts/FindCBLAS.cmake

# - Find CBLAS library
#
# This module finds an installed fortran library that implements the CBLAS 
# linear-algebra interface (see http://www.netlib.org/blas/), with CBLAS
# interface.
#
# This module sets the following variables:
#  CBLAS_FOUND - set to true if a library implementing the CBLAS interface is found
#  CBLAS_LIBRARIES - list of libraries (using full path name) to link against to use CBLAS
#  CBLAS_INCLUDE_DIR - path to includes
#  CBLAS_INCLUDE_FILE - the file to be included to use CBLAS
#

SET(CBLAS_LIBRARIES)
SET(CBLAS_INCLUDE_DIR)
SET(CBLAS_INCLUDE_FILE)

# CBLAS in Intel MKL
FIND_PACKAGE(MKL)
IF (MKL_FOUND AND NOT CBLAS_LIBRARIES)
  SET(CBLAS_LIBRARIES ${MKL_LIBRARIES})
  SET(CBLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  SET(CBLAS_INCLUDE_FILE "mkl_cblas.h")
  RETURN ()
ENDIF (MKL_FOUND AND NOT CBLAS_LIBRARIES)

# Old CBLAS search 
INCLUDE(CheckLibraryList)

# Apple CBLAS library?
IF(NOT CBLAS_LIBRARIES)
  CHECK_LIBRARY_LIST(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "Accelerate"
    "Accelerate/Accelerate.h"
    TRUE )
ENDIF(NOT CBLAS_LIBRARIES)

IF( NOT CBLAS_LIBRARIES )
  CHECK_LIBRARY_LIST(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "vecLib"
    "vecLib/vecLib.h"
    TRUE )
ENDIF( NOT CBLAS_LIBRARIES )

# CBLAS in ATLAS library? (http://math-atlas.sourceforge.net/)
IF(NOT CBLAS_LIBRARIES)
  CHECK_LIBRARY_LIST(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "cblas;atlas"
    "cblas.h"
    TRUE )
ENDIF(NOT CBLAS_LIBRARIES)

# Generic CBLAS library
IF(NOT CBLAS_LIBRARIES)
  CHECK_LIBRARY_LIST(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "cblas"
    "cblas.h"
    TRUE )
ENDIF(NOT CBLAS_LIBRARIES)

# CBLAS library included in BLAS library
IF(NOT CBLAS_LIBRARIES)
  CHECK_LIBRARY_LIST(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "blas"
    "cblas.h"
    TRUE )
ENDIF(NOT CBLAS_LIBRARIES)

IF(CBLAS_LIBRARIES)
  SET(CBLAS_FOUND TRUE)
ELSE(CBLAS_LIBRARIES)
  SET(CBLAS_FOUND FALSE)
ENDIF(CBLAS_LIBRARIES)

IF(NOT CBLAS_FOUND AND CBLAS_FIND_REQUIRED)
  MESSAGE(FATAL_ERROR "CBLAS library not found. Please specify library location")
ENDIF(NOT CBLAS_FOUND AND CBLAS_FIND_REQUIRED)
IF(NOT CBLAS_FIND_QUIETLY)
  IF(CBLAS_FOUND)
    MESSAGE(STATUS "CBLAS library found")
  ELSE(CBLAS_FOUND)
    MESSAGE(STATUS "CBLAS library not found.")
  ENDIF(CBLAS_FOUND)
ENDIF(NOT CBLAS_FIND_QUIETLY)