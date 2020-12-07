#ifndef __BTAS_SCAL_IMPL_H
#define __BTAS_SCAL_IMPL_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <btas/tensor_traits.h>
#include <btas/types.h>
#include <btas/type_traits.h>

#include <btas/generic/numeric_type.h>
#include <btas/generic/tensor_iterator_wrapper.h>
#include <btas/generic/blas_lapack_delegator.h>

namespace btas {

//  ================================================================================================

/// Call BLAS depending on type of Tensor class
template<bool _Finalize> struct scal_impl { };

/// Case that alpha is trivially multipliable to elements
template<> struct scal_impl<true>
{
   template<typename _T, class _IteratorX>
   static void call_impl (
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            generic_impl_tag)
   {
      for (unsigned long i = 0; i < Nsize; ++i, itrX += incX)
      {
         (*itrX) *= alpha;
      }
   }

#ifdef BTAS_HAS_BLAS_LAPACK
   template<typename _T, class _IteratorX>
   static void call_impl (
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX,
            blas_lapack_impl_tag)
   {
     //std::cout << "IN BLASPP SCAL IMPL" << std::endl;
     blas::scal( Nsize, alpha, static_cast<_T*>(&(*itrX)), incX );
   }
#endif

   template<typename _T, class _IteratorX>
   static void call (
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX )
   {
     call_impl( Nsize, alpha, itrX, incX, blas_lapack_impl_t<_IteratorX>() );
   }

};

/// Case that alpha is multiplied recursively by SCAL
/// Note that incX is disabled for recursive call
template<> struct scal_impl<false>
{
   template<typename _T, class _IteratorX>
   static void call (
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX)
   {
      for (unsigned long i = 0; i < Nsize; ++i, itrX += incX)
      {
         scal(alpha, *itrX);
      }
   }
};

//  ================================================================================================

/// Generic implementation of BLAS SCAL in terms of C++ iterator
template<typename _T, class _IteratorX>
void scal (
   const unsigned long& Nsize,
   const _T& alpha,
         _IteratorX itrX, const iterator_difference_t<_IteratorX>& incX)
{
   typedef std::iterator_traits<_IteratorX> __traits_X;

   static_assert(is_random_access_iterator_v<_IteratorX>, "iterator X must be a random access iterator");

   typedef typename __traits_X::value_type __value_X;
   typedef typename std::conditional<std::is_convertible<_T, __value_X>::value, __value_X, _T>::type __alpha;
   scal_impl<std::is_convertible<_T, __value_X>::value>::call(Nsize, static_cast<__alpha>(alpha), itrX, incX);
}

//  ================================================================================================

/// Convenient wrapper to call BLAS SCAL from tensor objects
template<
   typename _T,
   class _TensorX,
   class = typename std::enable_if<
      is_boxtensor<_TensorX>::value
   >::type
>
void scal (
   const _T& alpha,
         _TensorX& X)
{
   if (X.empty())
   {
      return;
   }

   auto itrX = std::begin(X);

   scal (X.size(), alpha, itrX, 1);
}

} // namespace btas

#endif // __BTAS_SCAL_IMPL_H
