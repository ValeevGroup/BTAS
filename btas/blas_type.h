#ifndef __BTAS_BLAS_TYPE_H
#define __BTAS_BLAS_TYPE_H

#ifdef _HAS_BLAS
extern "C"
{
#ifdef _HAS_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
}

namespace btas
{

template <typename T>
struct BLAS_TYPE
    { };

template <>
struct BLAS_TYPE<double>
    {
        const double ZERO = 0.0;
        const double ONE  = 1.0;
        inline void COPY(const size_t& n, const double* x, const int& incx, double* y, const int& incy)
        {
            cblas_dxopy(n, x, incx, y, incy);
        }

        inline void AXPY(const size_t& n, const double& alpha, const double* x, const int& incx, double* y, const int& incy)
        {
            cblas_daxpy(n, alpha, x, incx, y, incy);
        }
    };

}; // namespace btas

#endif
