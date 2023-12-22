#ifndef __BTAS_GER_IMPL_H
#define __BTAS_GER_IMPL_H 1

#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>

#include <btas/tensor_traits.h>
#include <btas/types.h>
#include <btas/array_adaptor.h>
#include <btas/type_traits.h>

#include <btas/generic/numeric_type.h>
#include <btas/generic/tensor_iterator_wrapper.h>
#include <btas/generic/blas_lapack_delegator.h>

namespace btas {

template<bool _Finalize> struct ger_impl { };

template<> struct ger_impl<true>
{
   /// Performs GER operation
   template<typename _T, class _IteratorX, class _IteratorY, class _IteratorA>
   static void call_impl (
      const blas::Layout& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX,
      const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY,
      const iterator_difference_t<_IteratorY>& incY,
            _IteratorA itrA,
      const unsigned long& LDA,
      generic_impl_tag)
   {
      // RowMajor
      if (order == blas::Layout::RowMajor)
      {
         auto itrY_save = itrY;
         for (size_type i = 0; i < Msize; ++i, ++itrX)
         {
            itrY = itrY_save;
            for (size_type j = 0; j < Nsize; ++j, ++itrY, ++itrA)
            {
               (*itrA) += alpha * (*itrX) * (*itrY);
            }
         }
      }
      // A: ColMajor
      else
      {
         auto itrX_save = itrX;
         for (size_type i = 0; i < Nsize; ++i, ++itrY)
         {
            itrX = itrX_save;
            for (size_type j = 0; j < Msize; ++j, ++itrX, ++itrA)
            {
               (*itrA) += alpha * (*itrX) * (*itrY);
            }
         }
      }
   }

#ifdef BTAS_HAS_BLAS_LAPACK
   template<typename _T, class _IteratorX, class _IteratorY, class _IteratorA>
   static void call_impl (
      const blas::Layout& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX,
      const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY,
      const iterator_difference_t<_IteratorY>& incY,
            _IteratorA itrA,
      const unsigned long& LDA,
      blas_lapack_impl_tag)
   {
     static_assert(std::is_same_v<iterator_value_t<_IteratorX>,iterator_value_t<_IteratorY>> &&
                   std::is_same_v<iterator_value_t<_IteratorX>,iterator_value_t<_IteratorA>>,
                   "mismatching iterator value types");
     using T = iterator_value_t<_IteratorX>;

     blas::geru( order, Msize, Nsize, static_cast<T>(alpha),
                 static_cast<const T*>(&(*itrX)), incX,
                 static_cast<const T*>(&(*itrY)), incY,
                 static_cast<      T*>(&*(itrA)), LDA );
   }
#endif

   template<typename _T, class _IteratorX, class _IteratorY, class _IteratorA>
   static void call (
      const blas::Layout& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX,
      const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY,
      const iterator_difference_t<_IteratorY>& incY,
            _IteratorA itrA,
      const unsigned long& LDA)
   {

     call_impl( order, Msize, Nsize, alpha, itrX, incX, itrY, incY, itrA, LDA,
                blas_lapack_impl_t<_IteratorX, _IteratorY, _IteratorA>() );

   }

};

template<> struct ger_impl<false>
{
   /// GER implementation
   template<typename _T, class _IteratorX, class _IteratorY, class _IteratorA>
   static void call (
      const blas::Layout& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX,
      const iterator_difference_t<_IteratorX>& incX,
            _IteratorY itrY,
      const iterator_difference_t<_IteratorY>& incY,
            _IteratorA itrA,
      const unsigned long& LDA)
   {
      // RowMajor
      if (order == blas::Layout::RowMajor)
      {
         auto itrY_save = itrY;
         for (size_type i = 0; i < Msize; ++i, ++itrX)
         {
            itrY = itrY_save;
            for (size_type j = 0; j < Nsize; ++j, ++itrY, ++itrA)
            {
               ger(order, alpha, *itrX, *itrY, *itrA);
            }
         }
      }
      // A: ColMajor
      else
      {
         auto itrX_save = itrX;
         for (size_type i = 0; i < Nsize; ++i, ++itrY)
         {
            itrX = itrX_save;
            for (size_type j = 0; j < Msize; ++j, ++itrX, ++itrA)
            {
               ger(order, alpha, *itrX, *itrY, *itrA);
            }
         }
      }
   }

};

//  ================================================================================================

/// Generic implementation of BLAS GER in terms of C++ iterator
template<typename _T, class _IteratorX, class _IteratorY, class _IteratorA>
void ger (
   const blas::Layout& order,
   const unsigned long& Msize,
   const unsigned long& Nsize,
   const _T& alpha,
         _IteratorX itrX,
   const iterator_difference_t<_IteratorX>& incX,
         _IteratorY itrY,
   const iterator_difference_t<_IteratorY>& incY,
         _IteratorA itrA,
   const unsigned long& LDA)
{
   typedef std::iterator_traits<_IteratorX> __traits_X;
   typedef std::iterator_traits<_IteratorY> __traits_Y;
   typedef std::iterator_traits<_IteratorA> __traits_A;

   typedef typename __traits_A::value_type value_type;

   static_assert(std::is_same<value_type, typename __traits_X::value_type>::value, "value type of X must be the same as that of A");
   static_assert(std::is_same<value_type, typename __traits_Y::value_type>::value, "value type of Y must be the same as that of A");

   static_assert(std::is_same<typename __traits_X::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator X must be a random access iterator");

   static_assert(std::is_same<typename __traits_Y::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator Y must be a random access iterator");

   static_assert(std::is_same<typename __traits_A::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator A must be a random access iterator");

   typename __traits_X::pointer X = &(*itrX);
   typename __traits_Y::pointer Y = &(*itrY);
   typename __traits_A::pointer A = &(*itrA);
   ger_impl<std::is_convertible<_T, value_type>::value>::call(order, Msize, Nsize, alpha, X, incX, Y, incY, A, LDA);
}

//  ================================================================================================

/// Generic implementation of operation (generalization of BLAS GER operation)
/// \param alpha scalar value to be multiplied to A * X * Y
/// \param X input tensor
/// \param Y input tensor
/// \param A output tensor which can be empty tensor but needs to have rank info (= size of shape).
/// Iterator is assumed to be consecutive (or, random_access_iterator) , thus e.g. iterator to map doesn't work.
template<
   typename _T,
   class _TensorX, class _TensorY, class _TensorA,
   class = typename std::enable_if<
      is_boxtensor<_TensorX>::value &
      is_boxtensor<_TensorY>::value &
      is_boxtensor<_TensorA>::value
   >::type
>
void ger (
   const _T& alpha,
   const _TensorX& X,
   const _TensorY& Y,
         _TensorA& A)
{
    static_assert(boxtensor_storage_order<_TensorX>::value == boxtensor_storage_order<_TensorA>::value &&
                  boxtensor_storage_order<_TensorY>::value == boxtensor_storage_order<_TensorA>::value,
                  "btas::ger does not support mixed storage order");
    static_assert(boxtensor_storage_order<_TensorY>::value != boxtensor_storage_order<_TensorY>::other,
                  "btas::ger does not support non-major storage order");
    const blas::Layout order = boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorA>::row_major ?
                              blas::Layout::RowMajor : blas::Layout::ColMajor;

   if (X.empty() || Y.empty())
   {
      return;
   }

   // get contraction rank
   const size_type rankX = rank(X);
   const size_type rankY = rank(Y);

   // get shapes
   const typename _TensorX::range_type::extent_type& extentX = extent(X);
   const typename _TensorY::range_type::extent_type& extentY = extent(Y);
         typename _TensorA::range_type::extent_type  extentA;
   if (!A.empty())
     extentA = extent(A);
   else {
     array_adaptor<decltype(extentA)>::resize(extentA, rank(extentX) + rank(extentY));
     std::copy(std::begin(extentX), std::end(extentX), std::begin(extentA));
     std::copy(std::begin(extentY), std::end(extentY), std::begin(extentA) + rank(extentX));
   }


   size_type Msize = std::accumulate(std::begin(extentX), std::end(extentX), 1ul, std::multiplies<size_type>());
   size_type Nsize = std::accumulate(std::begin(extentY), std::end(extentY), 1ul, std::multiplies<size_type>());
   size_type LDA   = (order == blas::Layout::RowMajor) ? Nsize : Msize;

   std::copy_n(std::begin(extentX), rankX, std::begin(extentA));
   std::copy_n(std::begin(extentY), rankY, std::begin(extentA)+rankX);

   // resize / scale
   if (A.empty())
   {
     typedef typename _TensorA::value_type value_type;
     A.resize(extentA);
     NumericType<value_type>::fill(std::begin(A), std::end(A), NumericType<value_type>::zero());
   }
   else
   {
      assert(std::equal(std::begin(extentA), std::end(extentA), std::begin(extent(A))));
   }

   auto itrX = std::begin(X);
   auto itrY = std::begin(Y);
   auto itrA = std::begin(A);

   ger (order, Msize, Nsize, alpha, itrX, 1, itrY, 1, itrA, LDA);
}

} // namespace btas

#endif // __BTAS_GER_IMPL_H
