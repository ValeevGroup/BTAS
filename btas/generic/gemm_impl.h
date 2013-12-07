#ifndef __BTAS_GEMM_IMPL_H
#define __BTAS_GEMM_IMPL_H 1

#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>

#include <btas/tensor_traits.h>
#include <btas/types.h>

#include <btas/generic/numerictype.h>

namespace btas {

template<bool _DoGemm> struct gemm_impl { };

template<> struct gemm_impl<true>
{
   typedef unsigned long size_type;

   template<typename _T, class _IteratorA, class _IteratorB, class _IteratorC>
   gemm_impl (
         CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
         const size_type& Msize, const size_type& Nsize, const size_type& Ksize,
         const _T& alpha, _IteratorA beginA, _IteratorB beginB, const _T& beta, _IteratorC beginC)
   {
      // check iterator category
      static_assert(std::is_same<typename std::iterator_traits<_IteratorA>::iterator_category, std::random_access_iterator_tag>::value, "iterator A must be a random access iterator");
      static_assert(std::is_same<typename std::iterator_traits<_IteratorB>::iterator_category, std::random_access_iterator_tag>::value, "iterator B must be a random access iterator");
      static_assert(std::is_same<typename std::iterator_traits<_IteratorC>::iterator_category, std::random_access_iterator_tag>::value, "iterator C must be a random access iterator");

      // A:NoTrans / B:NoTrans
      if (transA == CblasNoTrans && transB == CblasNoTrans)
      {
         auto itrA      = beginA;
         auto itrC_save = beginC;
         for (size_type i = 0; i < Msize; ++i)
         {
            auto itrB = beginB;
            for (size_type k = 0; k < Ksize; ++k, ++itrA)
            {
               auto itrC = itrC_save;
               for (size_type j = 0; j < Nsize; ++j, ++itrB, ++itrC)
               {
                  (*itrC) += alpha * (*itrA) * (*itrB);
               }
            }
            itrC_save += Nsize;
         }
      }
      // A:NoTrans / B:Trans
      else if (transA == CblasNoTrans && transB != CblasNoTrans)
      {
         auto itrA_save = beginA;
         auto itrC      = beginC;
         for (size_type i = 0; i < Nsize; ++i)
         {
            auto itrB = beginB;
            for (size_type j = 0; j < Msize; ++j, ++itrC)
            {
               auto itrA = itrA_save;
               for (size_type k = 0; k < Ksize; ++k, ++itrA, ++itrB)
               {
                  (*itrC) += alpha * (*itrA) * (*itrB);
               }
            }
            itrA_save += Ksize;
         }
      }
      // A:Trans / B:NoTrans
      else if (transA != CblasNoTrans && transB == CblasNoTrans)
      {
         auto itrA      = beginA;
         auto itrB_save = beginB;
         for (size_type k = 0; k < Ksize; ++k)
         {
            auto itrC = beginC;
            for (size_type i = 0; i < Msize; ++i, ++itrA)
            {
               auto itrB = itrB_save;
               for (size_type j = 0; j < Nsize; ++j, ++itrB, ++itrC)
               {
                  (*itrC) += alpha * (*itrA) * (*itrB);
               }
            }
            itrB_save += Nsize;
         }
      }
      // A:Trans / B:Trans
      else if (transA != CblasNoTrans && transB != CblasNoTrans)
      {
         auto itrB      = beginB;
         auto itrC_save = beginC;
         for (size_type j = 0; j < Nsize; ++j, ++itrC_save)
         {
            auto itrA = beginA;
            for (size_type k = 0; k < Ksize; ++k, ++itrB)
            {
               auto itrC = itrC_save;
               for (size_type i = 0; i < Msize; ++i, ++itrA, itrC += Nsize)
               {
                  (*itrC) += alpha * (*itrA) * (*itrB);
               }
            }
         }
      }
   }
};

template<> struct gemm_impl<false>
{
   typedef unsigned long size_type;

   template<typename _T, class _IteratorA, class _IteratorB, class _IteratorC>
   gemm_impl (
         const CBLAS_ORDER& order,
         const CBLAS_TRANSPOSE& transA,
         const CBLAS_TRANSPOSE& transB,
         const unsigned long& Msize,
         const unsigned long& Nsize,
         const unsigned long& Ksize,
         const _T& alpha,
               _IteratorA beginA, const typename std::iterator_traits<_IteratorA>::difference_type& ldA,
               _IteratorB beginB, const typename std::iterator_traits<_IteratorB>::difference_type& ldB,
         const _T& beta,
               _IteratorC beginC, const typename std::iterator_traits<_IteratorC>::difference_type& ldC)
   {
      // check iterator category
      static_assert(std::is_same<typename std::iterator_traits<_IteratorA>::iterator_category, std::random_access_iterator_tag>::value, "iterator A must be a random access iterator");
      static_assert(std::is_same<typename std::iterator_traits<_IteratorB>::iterator_category, std::random_access_iterator_tag>::value, "iterator B must be a random access iterator");
      static_assert(std::is_same<typename std::iterator_traits<_IteratorC>::iterator_category, std::random_access_iterator_tag>::value, "iterator C must be a random access iterator");

      if (order == CblasColMajor)
      {
         gemm_impl(CblasRowMajor, transB, transA, Nsize, Msize, Ksize, alpha, beginB, ldB, beginA, ldA, beta, beginC, ldC);
      }
      // A:NoTrans / B:NoTrans
      else if (transA == CblasNoTrans && transB == CblasNoTrans)
      {
         auto itrA      = beginA;
         auto itrC_save = beginC;
         for (size_type i = 0; i < Msize; ++i)
         {
            auto itrB = beginB;
            for (size_type k = 0; k < Ksize; ++k, ++itrA)
            {
               auto itrC = itrC_save;
               for (size_type j = 0; j < Nsize; ++j, ++itrB, ++itrC)
               {
                  gemm(order, transA, transB, alpha, *itrA, *itrB, beta, *itrC);
               }
            }
            itrC_save += Nsize;
         }
      }
      // A:NoTrans / B:Trans
      else if (transA == CblasNoTrans && transB != CblasNoTrans)
      {
         auto itrA_save = beginA;
         auto itrC      = beginC;
         for (size_type i = 0; i < Nsize; ++i)
         {
            auto itrB = beginB;
            for (size_type j = 0; j < Msize; ++j, ++itrC)
            {
               auto itrA = itrA_save;
               for (size_type k = 0; k < Ksize; ++k, ++itrA, ++itrB)
               {
                  gemm(order, transA, transB, alpha, *itrA, *itrB, beta, *itrC);
               }
            }
            itrA_save += Ksize;
         }
      }
      // A:Trans / B:NoTrans
      else if (transA != CblasNoTrans && transB == CblasNoTrans)
      {
         auto itrA      = beginA;
         auto itrB_save = beginB;
         for (size_type k = 0; k < Ksize; ++k)
         {
            auto itrC = beginC;
            for (size_type i = 0; i < Msize; ++i, ++itrA)
            {
               auto itrB = itrB_save;
               for (size_type j = 0; j < Nsize; ++j, ++itrB, ++itrC)
               {
                  gemm(order, transA, transB, alpha, *itrA, *itrB, beta, *itrC);
               }
            }
            itrB_save += Nsize;
         }
      }
      // A:Trans / B:Trans
      else if (transA != CblasNoTrans && transB != CblasNoTrans)
      {
         auto itrB      = beginB;
         auto itrC_save = beginC;
         for (size_type j = 0; j < Nsize; ++j, ++itrC_save)
         {
            auto itrA = beginA;
            for (size_type k = 0; k < Ksize; ++k, ++itrB)
            {
               auto itrC = itrC_save;
               for (size_type i = 0; i < Msize; ++i, ++itrA, itrC += Nsize)
               {
                  gemm(order, transA, transB, alpha, *itrA, *itrB, beta, *itrC);
               }
            }
         }
      }
   }
};

/// Generic implementation of BLAS-GEMM
/// \param order storage order of tensor in matrix view (CblasRowMajor, CblasColMajor)
/// \param transA transpose directive for tensor A (CblasNoTrans, CblasTrans, CblasConjTrans)
/// \param transB transpose directive for tensor B (CblasNoTrans, CblasTrans, CblasConjTrans)
/// \param alpha scalar value to be multiplied to A * B
/// \param A input tensor
/// \param B input tensor
/// \param beta scalar value to be multiplied to C
/// \param C output tensor which can be empty tensor but needs to have rank info (= size of shape).
/// Iterator is assumed to be consecutive (or, random_access_iterator) , thus e.g. iterator to map doesn't work.
template<
   typename _T,
   class _TensorA, class _TensorB, class _TensorC,
   class = typename std::enable_if<
      is_tensor<_TensorA>::value &
      is_tensor<_TensorB>::value &
      is_tensor<_TensorC>::value
   >::type
>
void gemm (
   const CBLAS_ORDER& order,
   const CBLAS_TRANSPOSE& transA,
   const CBLAS_TRANSPOSE& transB,
   const _T& alpha,
   const _TensorA& a,
   const _TensorB& b,
   const _T& beta,
         _TensorC& c)
{
   typedef unsigned long size_type;

   // check element type
   static_assert(std::is_same<typename _TensorA::value_type, typename _TensorB::value_type>::value, "value type of B must be the same as that of A");
   static_assert(std::is_same<typename _TensorA::value_type, typename _TensorC::value_type>::value, "value type of C must be the same as that of A");

   if (a.empty() || b.empty()) return;

   // get contraction rank
   const size_type rankA = a.rank();
   const size_type rankB = b.rank();
   const size_type rankC = c.rank();

   const size_type K = (rankA+rankB-rankC)/2;
   const size_type M = rankA-K;
   const size_type N = rankB-K;

   // get shapes
   const typename _TensorA::shape_type& shapeA = a.shape();
   const typename _TensorB::shape_type& shapeB = b.shape();
         typename _TensorC::shape_type  shapeC = c.shape(); // if C is empty, this gives { 0,...,0 }

   size_type Msize = 0; // Rows count of C
   size_type Nsize = 0; // Cols count of C
   size_type Ksize = 0; // Dims to be contracted

   if (transA == CblasNoTrans)
   {
      Msize = std::accumulate(shapeA.begin(), shapeA.begin()+M, 1ul, std::multiplies<size_type>());
      Ksize = std::accumulate(shapeA.begin()+M, shapeA.end(),   1ul, std::multiplies<size_type>());

      for (size_type i = 0; i < M; ++i) shapeC[i] = shapeA[i];

      if (transB == CblasNoTrans)
         assert(std::equal(shapeA.begin()+M, shapeA.end(), shapeB.begin()));
      else
         assert(std::equal(shapeA.begin()+M, shapeA.end(), shapeB.begin()));
   }
   else
   {
      Msize = std::accumulate(shapeA.begin()+K, shapeA.end(),   1ul, std::multiplies<size_type>());
      Ksize = std::accumulate(shapeA.begin(), shapeA.begin()+K, 1ul, std::multiplies<size_type>());

      for (size_type i = 0; i < M; ++i) shapeC[i] = shapeA[K+i];

      if (transB == CblasNoTrans)
         assert(std::equal(shapeA.begin()+M, shapeA.end(), shapeB.begin()+N));
      else
         assert(std::equal(shapeA.begin()+M, shapeA.end(), shapeB.begin()+N));
   }

   if (transB == CblasNoTrans)
   {
      Nsize = std::accumulate(shapeB.begin()+K, shapeB.end(),   1ul, std::multiplies<size_type>());
      for (size_type i = 0; i < N; ++i) shapeC[M+i] = shapeB[K+i];
   }
   else
   {
      Nsize = std::accumulate(shapeB.begin(), shapeB.begin()+N, 1ul, std::multiplies<size_type>());
      for (size_type i = 0; i < N; ++i) shapeC[M+i] = shapeB[i];
   }

   typedef typename std::iterator_traits<typename _TensorA::iterator>::value_type value_type;

   // resize / scale
   if (c.empty())
   {
      c.resize(shapeC);
      NumericType<value_type>::fill(c.begin(), c.end(), NumericType<value_type>::zero());
   }
   else
   {
      assert(std::equal(shapeC.begin(), shapeC.end(), c.shape().begin()));
      NumericType<value_type>::scal(c.begin(), c.end(), beta);
   }

   // Call GEMM depending on value type
   gemm_impl<std::is_convertible<_T, value_type>::value> call(transA, transB, Msize, Nsize, Ksize, alpha, a.begin(), b.begin(), beta, c.begin());
}

} // namespace btas

#endif // __BTAS_GEMM_IMPL_H
