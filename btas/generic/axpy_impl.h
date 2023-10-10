#ifndef __BTAS_AXPY_IMPL_H
#define __BTAS_AXPY_IMPL_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <btas/tensor.h>
#include <btas/tensor_traits.h>
#include <btas/types.h>
#include <btas/type_traits.h>

#include <btas/generic/numeric_type.h>
#include <btas/generic/tensor_iterator_wrapper.h>
#include <btas/generic/blas_lapack_delegator.h>


namespace btas {



//  ================================================================================================

/// Call BLAS depending on type of Tensor class
template<bool _Finalize> struct axpy_impl { };

/// Case that alpha is trivially multipliable to elements
template<> struct axpy_impl<true>
{

   template<typename _T, class _IteratorX, class _IteratorY>
   static void call_impl (
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY,
      generic_impl_tag)
   {

      for (unsigned long i = 0; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         (*itrY) += alpha * (*itrX);
      }
   }

#ifdef BTAS_HAS_BLAS_LAPACK
   template<typename _T, class _IteratorX, class _IteratorY>
   static void call_impl (
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY,
      blas_lapack_impl_tag)
   { 
      static_assert(std::is_same_v<iterator_value_t<_IteratorX>,iterator_value_t<_IteratorY>>,
                    "mismatching iterator value types");
      using T = iterator_value_t<_IteratorX>;

     blas::axpy( Nsize, static_cast<T>(alpha), static_cast<const T*>(&(*itrX)), incX,
                               static_cast<T*>(&(*itrY)),       incY );
   }
#endif

   template<typename _T, class _IteratorX, class _IteratorY>
   static void call (
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY)
   {

     call_impl( Nsize, alpha, itrX, incX, itrY, incY, 
                blas_lapack_impl_t<_IteratorX,_IteratorY>() );

   }

};

/// Case that alpha is multiplied recursively by AXPY
/// Note that incX and incY are disabled for recursive call
template<> struct axpy_impl<false>
{
   template<typename _T, class _IteratorX, class _IteratorY>
   static void call (
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY)
   {
      for (unsigned long i = 0; i < Nsize; ++i, itrX += incX, itrY += incY)
      {
         axpy(alpha, *itrX, *itrY);
      }
   }
};

//  ================================================================================================

/// Generic implementation of BLAS AXPY in terms of C++ iterator
template<typename _T, class _IteratorX, class _IteratorY>
void axpy (
   const unsigned long& Nsize,
   const _T& alpha,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY, const iterator_difference_t<_IteratorY>& incY)
{
   typedef std::iterator_traits<_IteratorX> __traits_X;
   typedef std::iterator_traits<_IteratorY> __traits_Y;

   static_assert(std::is_same<typename __traits_X::value_type, typename __traits_Y::value_type>::value, "value type of Y must be the same as that of X");
   static_assert(is_random_access_iterator_v<_IteratorX>, "iterator X must be a random access iterator");
   static_assert(is_random_access_iterator_v<_IteratorY>, "iterator Y must be a random access iterator");

   typedef typename __traits_X::value_type __value_X;
   typedef typename std::conditional<std::is_convertible<_T, __value_X>::value, __value_X, _T>::type __alpha;
   axpy_impl<std::is_convertible<_T, __value_X>::value>::call(Nsize, static_cast<__alpha>(alpha), itrX, incX, itrY, incY);
}

//  ================================================================================================

/// Convenient wrapper to call BLAS AXPY from tensor objects
template<
   typename _T,
   class _TensorX, class _TensorY,
   class = typename std::enable_if<
      is_boxtensor<_TensorX>::value &
      is_boxtensor<_TensorY>::value
   >::type
>
void axpy (
   const _T& alpha,
   const _TensorX& X,
         _TensorY& Y)
{
   typedef typename _TensorX::value_type value_type;
   static_assert(std::is_same<value_type, typename _TensorY::value_type>::value, "value type of Y must be the same as that of X");

   if (X.empty())
   {
      Y.clear();
      return;
   }

   if (Y.empty())
   {
      Y.resize(btas::extent(X));
      NumericType<value_type>::fill(std::begin(Y), std::end(Y), NumericType<value_type>::zero());
   }
   else
   {
      assert( range(X) == range(Y) );
   }

   auto itrX = std::begin(X);
   auto itrY = std::begin(Y);

   axpy (X.size(), alpha, itrX, 1, itrY, 1);
}

} // namespace btas

#endif // __BTAS_AXPY_IMPL_H
