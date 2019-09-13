#ifndef __BTAS_TYPES_H
#define __BTAS_TYPES_H 1

//
//  BLAS types
//

#include <complex>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef BTAS_HAS_CBLAS

#if not defined(_CBLAS_HEADER) && not defined(_LAPACKE_HEADER)

#ifdef _HAS_INTEL_MKL

#include <mkl_cblas.h>
#include <mkl_lapacke.h>

#else  // _HAS_INTEL_MKL

#include <cblas.h>
#include <lapacke.h>

#endif  // _HAS_INTEL_MKL

#else  // _CBLAS_HEADER

#include _CBLAS_HEADER
#include _LAPACKE_HEADER

#endif  // _CBLAS_HEADER

#else  // BTAS_HAS_CBLAS

/// major order directive
enum CBLAS_ORDER { CblasRowMajor, CblasColMajor };

/// transposition directive in gemv and gemm
enum CBLAS_TRANSPOSE { CblasNoTrans, CblasTrans, CblasConjTrans };

/// upper / lower triangular directive (not used)
enum CBLAS_UPLO { CblasUpper, CblasLower };

/// diagonal matrix directive (not used)
enum CBLAS_DIAG { CblasNonUnit, CblasUnit };

/// transposition directive for symmetric matrix (not used)
enum CBLAS_SIDE { CblasLeft, CblasRight };

#endif // BTAS_HAS_CBLAS

#ifdef __cplusplus
}
#endif // __cplusplus

// some BLAS libraries define their own types for complex data
#ifndef HAVE_INTEL_MKL
#ifndef lapack_complex_float
# define lapack_complex_float  std::complex<float>
#else
static_assert(sizeof(std::complex<float>)==sizeof(lapack_complex_float), "sizes of lapack_complex_float and std::complex<float> do not match");
#endif
#ifndef lapack_complex_double
# define lapack_complex_double std::complex<double>
#else
static_assert(sizeof(std::complex<double>)==sizeof(lapack_complex_double), "sizes of lapack_complex_double and std::complex<double> do not match");
#endif
#else
// if calling direct need to cast to the MKL complex types
# ifdef MKL_DIRECT_CALL
#  include <mkl_types.h>
#  ifndef lapack_complex_float
#   define lapack_complex_float MKL_Complex8
#  endif
#  ifndef lapack_complex_double
#   define lapack_complex_double MKL_Complex16
#  endif
// else can call via F77 prototypes which don't need type conversion
# else
#  ifndef lapack_complex_float
#   define lapack_complex_float  std::complex<float>
#  endif
#  ifndef lapack_complex_double
#   define lapack_complex_double std::complex<double>
#  endif
# endif
#endif

namespace btas {

  template <typename T>
  T to_lapack_val(T val) {
    return val;
  }
  template <typename T>
  T from_lapack_val(T val) {
    return val;
  }

  inline lapack_complex_float to_lapack_val(std::complex<float> val) {
    return *reinterpret_cast<lapack_complex_float*>(&val);
  }
  inline std::complex<float> from_lapack_val(lapack_complex_float val) {
    return *reinterpret_cast<std::complex<float>*>(&val);
  }
  template <typename T>
  const lapack_complex_float*
  to_lapack_cptr(const T* ptr) {
    static_assert(sizeof(T)==sizeof(lapack_complex_float), "sizes of lapack_complex_float and T given to btas::to_lapack_cptr do not match");
    return reinterpret_cast<const lapack_complex_float*>(ptr);
  }
  template <typename T>
  typename std::enable_if<!std::is_const<T>::value, lapack_complex_float*>::type
  to_lapack_cptr(T* ptr) {
    static_assert(sizeof(T)==sizeof(lapack_complex_float), "sizes of lapack_complex_float and T given to btas::to_lapack_cptr do not match");
    return reinterpret_cast<lapack_complex_float*>(ptr);
  }

  inline lapack_complex_double to_lapack_val(std::complex<double> val) {
    return *reinterpret_cast<lapack_complex_double*>(&val);
  }
  inline std::complex<double> from_lapack_val(lapack_complex_double val) {
    return *reinterpret_cast<std::complex<double>*>(&val);
  }
  template <typename T>
  const lapack_complex_double*
  to_lapack_zptr(const T* ptr) {
    static_assert(sizeof(T)==sizeof(lapack_complex_double), "sizes of lapack_complex_double and T given to btas::to_lapack_zptr do not match");
    return reinterpret_cast<const lapack_complex_double*>(ptr);
  }
  template <typename T>
  typename std::enable_if<!std::is_const<T>::value, lapack_complex_double*>::type
  to_lapack_zptr(T* ptr) {
    static_assert(sizeof(T)==sizeof(lapack_complex_double), "sizes of lapack_complex_double and T given to btas::to_lapack_zptr do not match");
    return reinterpret_cast<lapack_complex_double*>(ptr);
  }

//
//  Other aliases for convenience
//

/// default size type
typedef unsigned long size_type;

/// null deleter
struct nulldeleter { void operator() (void const*) { } };

} // namespace btas

#endif // __BTAS_TYPES_H
