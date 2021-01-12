# BLAS++ / LAPACK++
if( BTAS_USE_BLAS_LAPACK )

  include(FetchContent)

  if(NOT TARGET blaspp)
    find_package( blaspp QUIET )
    if(NOT TARGET blaspp)
      cmake_minimum_required (VERSION 3.14.0)  # for FetchContent_MakeAvailable
      FetchContent_Declare( blaspp
            GIT_REPOSITORY https://bitbucket.org/icl/blaspp.git
            GIT_TAG        9097d6b61468e9e8899d0aed49af0371ca4fff1d
            )

      FetchContent_MakeAvailable( blaspp )
    endif(NOT TARGET blaspp)
  endif(NOT TARGET blaspp)

  if(NOT TARGET lapackpp)
    find_package( OpenMP QUIET ) #XXX Open LAPACKPP issue for this...
    find_package( lapackpp QUIET )
    if(NOT TARGET lapackpp)
      cmake_minimum_required (VERSION 3.14.0)  # for FetchContent_MakeAvailable
      FetchContent_Declare( lapackpp
            GIT_REPOSITORY https://bitbucket.org/icl/lapackpp.git
            GIT_TAG        213d13e584a8cb3bf0e7caf5d1d40cd10f4254ae
            )

      FetchContent_MakeAvailable( lapackpp )
    endif(NOT TARGET lapackpp)
  endif(NOT TARGET lapackpp)

  target_link_libraries( BTAS INTERFACE blaspp lapackpp )
  target_compile_definitions( BTAS INTERFACE -DBTAS_HAS_BLAS_LAPACK=1 -DLAPACK_COMPLEX_CPP=1 )

  ##################### Introspect BLAS/LAPACK libs

  # Check if BLAS/LAPACK is MKL
  set( BTAS_HAS_MKL )
  include( CheckFunctionExists )
  include(CMakePushCheckState)
  cmake_push_check_state( RESET )
  set( CMAKE_REQUIRED_LIBRARIES "${blaspp_libraries}" m )
  check_function_exists( mkl_dimatcopy  BLAS_IS_MKL )
  if( BLAS_IS_MKL )
    target_compile_definitions(BTAS INTERFACE -DBTAS_HAS_INTEL_MKL=1)
  endif(BLAS_IS_MKL)
  cmake_pop_check_state()

  # blaspp_header library is a target that permits #include'ing library-specific headers, e.g. mkl.h
  if (NOT TARGET blaspp_headers)

    add_library(blaspp_headers INTERFACE)

    if (BLAS_IS_MKL)
      foreach(_lib ${blaspp_libraries})
        if (EXISTS ${_lib} AND _lib MATCHES libmkl_)
          string(REGEX REPLACE "/lib/(intel64_lin/|intel64/|)libmkl_.*" "" _mklroot "${_lib}")
        elseif (_lib MATCHES "^-L")
          string(REGEX REPLACE "^-L" "" _mklroot "${_lib}")
          string(REGEX REPLACE "/lib(/intel64_lin|/intel64|)(/|)" "" _mklroot "${_mklroot}")
        endif()
        if (_mklroot)
          break()
        endif(_mklroot)
      endforeach()

      set(_mkl_include )
      if (EXISTS "${_mklroot}/include")
        set(_mkl_include "${_mklroot}/include")
      elseif(EXISTS "/usr/include/mkl") # ubuntu package
        set(_mkl_include "/usr/include/mkl")
      endif()
      if (_mkl_include AND EXISTS "${_mkl_include}")
        target_include_directories(blaspp_headers INTERFACE "${_mkl_include}")
      endif(_mkl_include AND EXISTS "${_mkl_include}")

    endif(BLAS_IS_MKL)

    install(TARGETS blaspp_headers EXPORT blaspp_headers)
    export(EXPORT blaspp_headers FILE "${PROJECT_BINARY_DIR}/blaspp_headers-targets.cmake")
    install(EXPORT blaspp_headers
            FILE "blaspp_headers-targets.cmake"
            DESTINATION "${BTAS_INSTALL_CMAKEDIR}")

  endif(NOT TARGET blaspp_headers)

  target_link_libraries( BTAS INTERFACE blaspp_headers )

endif( BTAS_USE_BLAS_LAPACK )
