#ifndef __BTAS_DOT_IMPL_H
#define __BTAS_DOT_IMPL_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <btas/tensor_traits.h>
#include <btas/types.h>

#include <btas/generic/numeric_type.h>
#include <btas/generic/tensor_iterator_wrapper.h>

namespace btas {

//  ================================================================================================

/// For general case
template<typename _T>
struct dot_impl
{
   typedef decltype(dot(_T, _T)) return_type;

   template<class _IteratorX, class _IteratorY>
   static return_type call (
      const unsigned long& Nsize,
            _IteratorX itrX, const typename std::iterator_traits<_IteratorX>::difference_type& incX,
            _IteratorY itrY, const typename std::iterator_traits<_IteratorY>::difference_type& incY)
   {
      return_type val = dot(*itrX, *itrY);
      for (unsigned long i = 1; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         val += dot(*itrX, *itrY);
      }
      return val;
   }
}

template<>
struct dot_impl<float>
{
   typedef float return_type;

   static return_type call (
      const unsigned long& Nsize,
      const float* itrX, const typename std::iterator_traits<float*>::difference_type& incX,
            float* itrY, const typename std::iterator_traits<float*>::difference_type& incY)
   {
#ifdef _HAS_CBLAS
      return cblas_sdot(Nsize, itrX, incX, itrY, incY);
#else
      return_type val = (*itrX) * (*itrY);
      for (unsigned long i = 1; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         val += (*itrX) * (*itrY);
      }
      return val;
#endif
   }
};

template<>
struct dot_impl<double>
{
   typedef double return_type;

   static return_type call (
      const unsigned long& Nsize,
      const double* itrX, const typename std::iterator_traits<double*>::difference_type& incX,
            double* itrY, const typename std::iterator_traits<double*>::difference_type& incY)
   {
#ifdef _HAS_CBLAS
      return cblas_sdot(Nsize, itrX, incX, itrY, incY);
#else
      return_type val = (*itrX) * (*itrY);
      for (unsigned long i = 1; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         val += (*itrX) * (*itrY);
      }
      return val;
#endif
   }
};

/// Case that alpha is multiplied recursively by DOT
/// Note that incX and incY are disabled for recursive call
template<> struct dot_impl<false>
{
   template<typename _T, class _IteratorX, class _IteratorY>
   static void call (
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX, const typename std::iterator_traits<_IteratorX>::difference_type& incX,
            _IteratorY itrY, const typename std::iterator_traits<_IteratorY>::difference_type& incY)
   {
      for (unsigned long i = 0; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         dot(alpha, *itrX, *itrY);
      }
   }
};

//  ================================================================================================

/// Generic implementation of BLAS DOT in terms of C++ iterator
template<typename _T, class _IteratorX, class _IteratorY>
auto dot (
   const unsigned long& Nsize,
         _IteratorX itrX, const typename std::iterator_traits<_IteratorX>::difference_type& incX,
         _IteratorY itrY, const typename std::iterator_traits<_IteratorY>::difference_type& incY)
   -> decltype(dot_impl<typename std::iterator_traits<_IteratorX>::value_type>::call(Nsize, itrX, incX, itrY, incY))
{
   typedef std::iterator_traits<_IteratorX> __traits_X;
   typedef std::iterator_traits<_IteratorY> __traits_Y;

   static_assert(std::is_same<typename __traits_X::value_type, typename __traits_Y::value_type>::value, "value type of Y must be the same as that of X");
   static_assert(std::is_same<typename __traits_X::iterator_category, std::random_access_iterator_tag>::value, "iterator X must be a random access iterator");
   static_assert(std::is_same<typename __traits_Y::iterator_category, std::random_access_iterator_tag>::value, "iterator Y must be a random access iterator");

   return dot_impl<typename __traits_X::value_type>::call(Nsize, itrX, incX, itrY, incY);
}

//  ================================================================================================

/// Convenient wrapper to call BLAS DOT from tensor objects
template<
   class _TensorX,
   class _TensorY,
   class = typename std::enable_if<
      is_tensor<_TensorX>::value &
      is_tensor<_TensorY>::value
   >::type
>
auto dot (
   const _TensorX& X,
         _TensorY& Y) -> decltype(dot(typename _Tensor::size_type, typename _Tensor::iterator, 1, typename _Tensor::iterator, 1))
{
   typedef typename _TensorX::value_type value_type;
   static_assert(std::is_same<value_type, typename _TensorY::value_type>::value, "value type of Y must be the same as that of X");

   if (X.empty() || Y.empty())
   {
      return 0;
   }

   auto itrX = tbegin(X);
   auto itrY = tbegin(Y);

   return dot(X.size(), itrX, 1, itrY, 1);
}

} // namespace btas

#endif // __BTAS_DOT_IMPL_H
