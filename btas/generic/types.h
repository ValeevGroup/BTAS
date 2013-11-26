#ifndef __BTAS_CBLAS_TYPES_H
#define __BTAS_CBLAS_TYPES_H 1

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef _HAS_CBLAS
#ifdef _HAS_INTEL_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif // _HAS_INTEL_MKL
#else

enum CBLAS_ORDER { CblasRowMajor, CblasColMajor };
enum CBLAS_TRANSPOSE { CblasNoTrans, CblasTrans, CblasConjTrans };
enum CBLAS_UPLO { CblasUpper, CblasLower };
enum CBLAS_DIAG { CblasNonUnit, CblasUnit };
enum CBLAS_SIDE { CblasLeft, CblasRight };

#endif // _HAS_CBLAS

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __BTAS_CBLAS_TYPES_H
