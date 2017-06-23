#ifndef __BTAS_TYPES_H
#define __BTAS_TYPES_H 1

//
//  BLAS types
//

#include <complex>
#define lapack_complex_float  std::complex<float>
#define lapack_complex_double std::complex<double>

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

namespace btas {

//
//  Other aliases for convenience
//

/// default size type
typedef unsigned long size_type;

/// null deleter
struct nulldeleter { void operator() (void const*) { } };

} // namespace btas

#endif // __BTAS_TYPES_H
