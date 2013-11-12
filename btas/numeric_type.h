#ifndef __BTAS_NUMERIC_TYPE_H
#define __BTAS_NUMERIC_TYPE_H 1

#include <complex>

#ifdef _HAS_CBLAS

#ifdef _HAS_INTEL_MKL

#include <mkl_cblas.h>
#define INTEGER_TYPE MKL_INT

#else

#include <cblas.h>
#define INTEGER_TYPE long

#endif // _HAS_INTEL_MKL

#else

#define INTEGER_TYPE unsigned long

#endif // _HAS_CBLAS

namespace btas {

template <typename T>
struct NUMERIC_TYPE
{
   constexpr static T zero () { static_assert (false, "Template parameter is not numeric type!"); return T(); }
   constexpr static T one  () { static_assert (false, "Template parameter is not numeric type!"); return T(); }
   constexpr static T two  () { static_assert (false, "Template parameter is not numeric type!"); return T(); }
// constexpr static T zero () { return static_cast<T>(0); }
// constexpr static T one  () { return static_cast<T>(1); }
// constexpr static T two  () { return static_cast<T>(2); }

   /// copy from x to y
   inline void copy  (INTEGER_TYPE n, T* x, T* y)
   {
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) *y = *x;
   }

   /// addition for each element
   /// addition assignment operator must be overloaded
   inline void plus  (INTEGER_TYPE n, double* x, double* y)
   {
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) *y += *x;
   }

   /// subtract for each element
   /// subtract assignment operator must be overloaded
   inline void minus (INTEGER_TYPE n, double* x, double* y)
   {
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) *y -= *x;
   }
};

//
// specialization for each numeric type goes from here
//

/// numeric type for double precision real number
template <>
struct NUMERIC_TYPE<double>
{
   /// \return 0
   constexpr static double zero () { return 0.0; }
   /// \return 1
   constexpr static double one  () { return 1.0; }
   /// \return 2
   constexpr static double two  () { return 2.0; }

   inline void copy  (INTEGER_TYPE n, double* x, double* y)
   {
#ifdef _HAS_CBLAS
      cblas_dcopy(n, x, 1, y, 1);
#else
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) *y = *x;
#endif
   }

   inline void plus  (INTEGER_TYPE n, double* x, double* y)
   {
#ifdef _HAS_CBLAS
      cblas_daxpy(n, 1.0, x, 1, y, 1);
#else
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) *y += *x;
#endif
   }

   inline void minus (INTEGER_TYPE n, double* x, double* y)
   {
#ifdef _HAS_CBLAS
      cblas_daxpy(n,-1.0, x, 1, y, 1);
#else
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) *y -= *x;
#endif
   }

#ifdef _HAS_CBLAS

   inline double dot (INTEGER_TYPE n, double* x, int incx, double* y, int incy)
   { return cblas_ddot (n, x, incx, y, incy); }

   inline void copy (INTEGER_TYPE n, double* x, int incx, double* y, int incy)
   { cblas_dcopy (n, x, incx, y, incy); }

   inline void scal (INTEGER_TYPE n, double alpha, double* x, int incx)
   { cblas_dscal (n, alpha, x, incx); }

   inline void axpy (INTEGER_TYPE n, double alpha, double* x, int incx, double* y, int incy)
   { cblas_daxpy (n, alpha, x, incx, y, incy); }

   inline void gemv (const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa,
                     INTEGER_TYPE m, INTEGER_TYPE n, double alpha, double* a, INTEGER_TYPE lda, double* b, INTEGER_TYPE ldb, double beta, double* c, INTEGER_TYPE ldc)
   { cblas_dgemv (order, transa, m, n, alpha, a, lda, b, ldb, beta, c, ldc); }

   inline void ger  (const CBLAS_ORDER order,
                     INTEGER_TYPE m, INTEGER_TYPE n, double alpha, double* x, INTEGER_TYPE incx, double* y, INTEGER_TYPE incy, double* a, INTEGER_TYPE lda)
   { cblas_dger  (order, m, n, alpha, x, incx, y, incy, a, lda); }

   inline void gemm (const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                     INTEGER_TYPE m, INTEGER_TYPE n, double alpha, double* a, INTEGER_TYPE lda, double* b, INTEGER_TYPE ldb, double beta, double* c, INTEGER_TYPE ldc)
   { cblas_dgemm (order, transa, transb, m, n, alpha, a, lda, b, ldb, beta, c, ldc); }

#else

   // TODO: write here non-BLAS implementations

#endif
};

/// numeric type for double precision complex number
template <>
struct NUMERIC_TYPE<std::complex<double>>
{
   /// \return 0
   constexpr static std::complex<double> zero () { return 0.0; }
   /// \return 1
   constexpr static std::complex<double> one  () { return 1.0; }
   /// \return 2
   constexpr static std::complex<double> two  () { return 2.0; }
   /// \return 1r this is safe for casting
   const     static std::complex<double> oner () { return std::complex<double>(1.0, 0.0); }
   /// \return 1i FIXME: can this be a 'constexpr'?
   const     static std::complex<double> onei () { return std::complex<double>(0.0, 1.0); }

   inline void copy  (INTEGER_TYPE n, std::complex<double>* x, std::complex<double>* y)
   {
#ifdef _HAS_CBLAS
      cblas_zcopy(n, x, 1, y, 1);
#else
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) *y = *x;
#endif
   }

   inline void plus  (INTEGER_TYPE n, std::complex<double>* x, std::complex<double>* y)
   {
#ifdef _HAS_CBLAS
      std::complex<double> alpha( 1.0, 0.0);
      cblas_zaxpy(n, &alpha, x, 1, y, 1);
#else
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) *y += *x;
#endif
   }

   inline void minus (INTEGER_TYPE n, std::complex<double>* x, std::complex<double>* y)
   {
#ifdef _HAS_CBLAS
      std::complex<double> alpha(-1.0, 0.0);
      cblas_zaxpy(n, &alpha, x, 1, y, 1);
#else
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) *y -= *x;
#endif
   }

#ifdef _HAS_CBLAS

   inline std::complex<double> dot (INTEGER_TYPE n, std::complex<double>* x, int incx, std::complex<double>* y, int incy)
   {
      std::complex<double> dotc;
      cblas_zdotc_sub (n, x, incx, y, incy, dotc);
      return dotc;
   }

   inline void copy (INTEGER_TYPE n, std::complex<double>* x, int incx, std::complex<double>* y, int incy)
   { cblas_zcopy (n, x, incx, y, incy); }

   inline void scal (INTEGER_TYPE n, std::complex<double> alpha, std::complex<double>* x, int incx)
   { cblas_zscal (n, &alpha, x, incx); }

   inline void axpy (INTEGER_TYPE n, std::complex<double> alpha, std::complex<double>* x, int incx, std::complex<double>* y, int incy)
   { cblas_zaxpy (n, &alpha, x, incx, y, incy); }

   inline void gemv (const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa,
                     INTEGER_TYPE m, INTEGER_TYPE n, std::complex<double> alpha, std::complex<double>* a, INTEGER_TYPE lda, std::complex<double>* b, INTEGER_TYPE ldb, std::complex<double> beta, std::complex<double>* c, INTEGER_TYPE ldc)
   { cblas_zgemv (order, transa, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc); }

   inline void ger  (const CBLAS_ORDER order,
                     INTEGER_TYPE m, INTEGER_TYPE n, std::complex<double> alpha, std::complex<double>* x, INTEGER_TYPE incx, std::complex<double>* y, INTEGER_TYPE incy, std::complex<double>* a, INTEGER_TYPE lda)
   { cblas_zger  (order, m, n, &alpha, x, incx, y, incy, a, lda); }

   inline void gemm (const CBLAS_ORDER order, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                     INTEGER_TYPE m, INTEGER_TYPE n, std::complex<double> alpha, std::complex<double>* a, INTEGER_TYPE lda, std::complex<double>* b, INTEGER_TYPE ldb, std::complex<double> beta, std::complex<double>* c, INTEGER_TYPE ldc)
   { cblas_zgemm (order, transa, transb, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc); }

#else

   // TODO: write here non-BLAS implementations

#endif
};

}; // namespace btas

#endif // __BTAS_NUMERIC_TYPE_H
