#ifndef __BTAS_AXPY_H
#define __BTAS_AXPY_H 1

namespace btas {

/// wrapper function
template<typename _T, class _Iterator>
void btas_axpy (const INTEGER_TYPE& n, const _T& alpha, _Iterator x, _Iterator y)
{
   btas_axpy_imple<std::is_pointer<_Iterator>::value, std::is_same<std::iterator_traits<_Iterator>::value_type, _T>::value> call(n, alpha, x, y);
}

/// class-like wrapper
template<bool _Consecutive, bool _Multipliable> struct btas_axpy_impl { };

/// case 1: _Iterator is consecutive & _Iterator::value_type == _T
template<>
struct btas_axpy_impl<true, true>
{
   /// axpy implementation
   template<typename _T>
   static void call (const INTEGER_TYPE& n, const _T& alpha, _T* x, _T* y)
   {
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) { (*y) += alpha * (*x); }
   }

   /// specialize for BLAS (real single prec.)
   template<>
   static void call<float>
   (const INTEGER_TYPE& n, const float& alpha, float* x, float* y)
   { cblas_saxpy(n, alpha, x, 1, y, 1); }

   /// specialize for BLAS (real double prec.)
   template<>
   static void call<double>
   (const INTEGER_TYPE& n, const double& alpha, double* x, double* y)
   { cblas_daxpy(n, alpha, x, 1, y, 1); }

   /// specialize for BLAS (complex single prec.)
   template<>
   static void call<std::complex<float>>
   (const INTEGER_TYPE& n, const std::complex<float>& alpha, std::complex<float>* x, std::complex<float>* y)
   { cblas_caxpy(n, alpha, x, 1, y, 1); }

   /// specialize for BLAS (complex double prec.)
   template<>
   static void call<std::complex<double>>
   (const INTEGER_TYPE& n, const std::complex<double>& alpha, std::complex<double>* x, std::complex<double>* y)
   { cblas_zaxpy(n, alpha, x, 1, y, 1); }
};

/// case 2: _Iterator is consecutive & _Iterator::value_type != _T (i.e. _T is an another Tensor class)
template<>
struct btas_axpy_impl<true, false>
{
   /// recursive call
   template<typename _T, class _Iterator>
   static void call (const INTEGER_TYPE& n, const _T& alpha, _Iterator x, _Iterator y)
   {
      for (INTEGER_TYPE i = 0; i < n; ++i, ++x, ++y) { axpy(alpha, *x, *y); }
   }
};

/// case 1: _Iterator is not consecutive & _Iterator::value_type == _T
template<>
struct btas_axpy_impl<false, true>
{
   /// axpy implementation
   template<typename _T>
   static void call (const INTEGER_TYPE& n, const _T& alpha, _Iterator x, _Iterator y)
   {
      for (INTEGER_TYPE i = 0; i < n; ++i)
      {
         if (!x[i]) continue;
         y[i] += alpha * x[i];
      }
   }
};

template<>
struct btas_axpy_impl<false, false>
{
};

}; // namespace btas

#endif // __BTAS_AXPY_IMPL_H
