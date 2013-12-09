#ifndef __BTAS_GER_IMPL_H
#define __BTAS_GER_IMPL_H 1

#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>

#include <btas/tensor_traits.h>
#include <btas/types.h>

#include <btas/generic/numeric_type.h>
#include <btas/generic/tensor_iterator_wrapper.h>

namespace btas {

template<bool _Finalize> struct ger_impl { };

template<> struct ger_impl<true>
{
   /// GER implementation
   template<typename _T, class _IteratorX, class _IteratorY, class _IteratorA>
   static void call (
      const CBLAS_ORDER& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX,
      const typename std::iterator_traits<_IteratorX>::difference_type& incX,
            _IteratorY itrY,
      const typename std::iterator_traits<_IteratorY>::difference_type& incY,
            _IteratorA itrA,
      const unsigned long& LDA)
   {
      // RowMajor
      if (order == CblasRowMajor)
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

#ifdef _HAS_CBLAS

   static void call (
      const CBLAS_ORDER& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const float& alpha,
      const float* itrX,
      const typename std::iterator_traits<float*>::difference_type& incX,
            float* itrY,
      const typename std::iterator_traits<float*>::difference_type& incY,
      const float* itrA,
      const unsigned long& LDA)
   {
      cblas_sger(order, Msize, Nsize, alpha, itrX, incX, itrY, incY, itrA, LDA);
   }

   static void call (
      const CBLAS_ORDER& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const double& alpha,
      const double* itrX,
      const typename std::iterator_traits<double*>::difference_type& incX,
            double* itrY,
      const typename std::iterator_traits<double*>::difference_type& incY,
      const double* itrA,
      const unsigned long& LDA)
   {
      cblas_dger(order, Msize, Nsize, alpha, itrX, incX, itrY, incY, itrA, LDA);
   }

   static void call (
      const CBLAS_ORDER& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const std::complex<float>& alpha,
      const std::complex<float>* itrX,
      const typename std::iterator_traits<std::complex<float>*>::difference_type& incX,
            std::complex<float>* itrY,
      const typename std::iterator_traits<std::complex<float>*>::difference_type& incY,
      const std::complex<float>* itrA,
      const unsigned long& LDA)
   {
      cblas_cger(order, Msize, Nsize, alpha, itrX, incX, itrY, incY, itrA, LDA);
   }

   static void call (
      const CBLAS_ORDER& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const std::complex<double>& alpha,
      const std::complex<double>* itrX,
      const typename std::iterator_traits<std::complex<double>*>::difference_type& incX,
            std::complex<double>* itrY,
      const typename std::iterator_traits<std::complex<double>*>::difference_type& incY,
      const std::complex<double>* itrA,
      const unsigned long& LDA)
   {
      cblas_zger(order, Msize, Nsize, alpha, itrX, incX, itrY, incY, itrA, LDA);
   }

#endif // _HAS_CBLAS

};

template<> struct ger_impl<false>
{
   /// GER implementation
   template<typename _T, class _IteratorX, class _IteratorY, class _IteratorA>
   static void call (
      const CBLAS_ORDER& order,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const _T& alpha,
            _IteratorX itrX,
      const typename std::iterator_traits<_IteratorX>::difference_type& incX,
            _IteratorY itrY,
      const typename std::iterator_traits<_IteratorY>::difference_type& incY,
            _IteratorA itrA,
      const unsigned long& LDA)
   {
      // RowMajor
      if (order == CblasRowMajor)
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
   const CBLAS_ORDER& order,
   const unsigned long& Msize,
   const unsigned long& Nsize,
   const _T& alpha,
         _IteratorX itrX,
   const typename std::iterator_traits<_IteratorX>::difference_type& incX,
         _IteratorY itrY,
   const typename std::iterator_traits<_IteratorY>::difference_type& incY,
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

   ger_impl<std::is_same<value_type, _T>::value>::call(order, Msize, Nsize, alpha, itrX, incX, itrY, incY, itrA, LDA);
}

//  ================================================================================================

/// Generic interface of BLAS-GER
/// \param order storage order of tensor in matrix view (CblasRowMajor, CblasColMajor)
/// \param alpha scalar value to be multiplied to A * X
/// \param X input tensor
/// \param Y input tensor
/// \param A output tensor which can be empty tensor but needs to have rank info (= size of shape).
/// Iterator is assumed to be consecutive (or, random_access_iterator) , thus e.g. iterator to map doesn't work.
template<
   typename _T,
   class _TensorX, class _TensorY, class _TensorA,
   class = typename std::enable_if<
      is_tensor<_TensorX>::value &
      is_tensor<_TensorY>::value &
      is_tensor<_TensorA>::value
   >::type
>
void ger (
   const CBLAS_ORDER& order,
   const _T& alpha,
   const _TensorX& X,
   const _TensorY& Y,
         _TensorA& A)
{
   // check element type
   typedef typename _TensorA::value_type value_type;
   static_assert(std::is_same<value_type, typename _TensorX::value_type>::value, "value type of X must be the same as that of A");
   static_assert(std::is_same<value_type, typename _TensorY::value_type>::value, "value type of Y must be the same as that of A");

   if (X.empty() || Y.empty())
   {
      return;
   }

   // get contraction rank
   const size_type rankX = X.rank();
   const size_type rankY = Y.rank();
   const size_type rankA = A.rank();

   // get shapes
   const typename _TensorX::shape_type& shapeX = X.shape();
   const typename _TensorY::shape_type  shapeY = Y.shape();
         typename _TensorA::shape_type& shapeA = A.shape();

   size_type Msize = std::accumulate(shapeX.begin(), shapeX.end(), 1ul, std::multiplies<size_type>());
   size_type Nsize = std::accumulate(shapeY.begin(), shapeY.end(), 1ul, std::multiplies<size_type>());
   size_type LDA   = (order == CblasRowMajor) ? Nsize : Msize;

   for (size_type i = 0; i < rankX; ++i) shapeA[i]       = shapeX[i];
   for (size_type i = 0; i < rankY; ++i) shapeA[i+rankX] = shapeY[i];

   // resize / scale
   if (A.empty())
   {
      A.resize(shapeA);
      NumericType<value_type>::fill(A.begin(), A.end(), NumericType<value_type>::zero());
   }
   else
   {
      assert(std::equal(shapeA.begin(), shapeA.end(), A.shape().begin()));
   }

   auto itrX = tbegin(X);
   auto itrY = tbegin(Y);
   auto itrA = tbegin(A);

   ger (order, Msize, Nsize, alpha, itrX, 1, itrY, 1, itrA, LDA);
}

} // namespace btas

#endif // __BTAS_GER_IMPL_H
