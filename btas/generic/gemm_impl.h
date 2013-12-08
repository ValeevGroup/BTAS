#ifndef __BTAS_GEMM_IMPL_H
#define __BTAS_GEMM_IMPL_H 1

#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>

#include <btas/tensor.h>
#include <btas/tensor_traits.h>
#include <btas/numerictype.h>
#include <btas/generic/types.h>
#include <btas/array_adaptor.h>

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
         CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
         const size_type& Msize, const size_type& Nsize, const size_type& Ksize,
         const _T& alpha, _IteratorA beginA, _IteratorB beginB, const _T& beta, _IteratorC beginC)
   {
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
                  gemm(transA, transB, alpha, *itrA, *itrB, beta, *itrC);
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
                  gemm(transA, transB, alpha, *itrA, *itrB, beta, *itrC);
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
                  gemm(transA, transB, alpha, *itrA, *itrB, beta, *itrC);
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
                  gemm(transA, transB, alpha, *itrA, *itrB, beta, *itrC);
               }
            }
         }
      }
   }
};

/// Generic implementation of BLAS-GEMM
/// \param transA transpose directive for tensor \param a (CblasNoTrans, CblasTrans, CblasConjTrans)
/// \param transB transpose directive for tensor \param b (CblasNoTrans, CblasTrans, CblasConjTrans)
/// \param alpha scalar value to be multiplied to \param a * \param b
/// \param a input tensor
/// \param b input tensor
/// \param beta scalar value to be multiplied to \param c
/// \param c output tensor which can be empty tensor but needs to have rank info
/// Iterator is assumed to be consecutive (or, random_access_iterator) , thus e.g. iterator to map doesn't work.
template<
   typename _T,
   class _TensorA, class _TensorB, class _TensorC,
   class = typename std::enable_if<
      is_boxtensor<_TensorA>::value &
      is_boxtensor<_TensorB>::value &
      is_boxtensor<_TensorC>::value &
      std::is_same<typename _TensorA::value_type, typename _TensorB::value_type>::value &
      std::is_same<typename _TensorA::value_type, typename _TensorC::value_type>::value &
      std::is_same<typename std::iterator_traits<typename _TensorA::iterator>::iterator_category, std::random_access_iterator_tag>::value &
      std::is_same<typename std::iterator_traits<typename _TensorB::iterator>::iterator_category, std::random_access_iterator_tag>::value &
      std::is_same<typename std::iterator_traits<typename _TensorC::iterator>::iterator_category, std::random_access_iterator_tag>::value
   >::type
>
void gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, const _T& alpha, const _TensorA& a, const _TensorB& b, const _T& beta, _TensorC& c)
{
   typedef unsigned long size_type;

   if (a.empty() || b.empty()) return;
   assert (c.rank() != 0);

// // check element types
// static_assert(std::is_same<typename _TensorA::value_type, typename _TensorB::value_type>::value, "type of A and B mismatches");
// static_assert(std::is_same<typename _TensorA::value_type, typename _TensorC::value_type>::value, "type of A and C mismatches");

   // only row-major BoxTensors are supported now
   static_assert(boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorA>::row_major &&
                 boxtensor_storage_order<_TensorB>::value == boxtensor_storage_order<_TensorB>::row_major &&
                 boxtensor_storage_order<_TensorC>::value == boxtensor_storage_order<_TensorC>::row_major
                 , "only row-major tensors are accepted right now");

   // get contraction rank
   const size_type rankA = a.rank();
   const size_type rankB = b.rank();
   const size_type rankC = c.rank();

   const size_type K = (rankA+rankB-rankC)/2;
   const size_type M = rankA-K;
   const size_type N = rankB-K;

   // get extents
   auto extentA = extent(a);
   auto extentB = extent(b);
   auto extentC = extent(c); // if C is empty, this gives { }, will need to allocate

   size_type Msize = 0; // Rows count of C
   size_type Nsize = 0; // Cols count of C
   size_type Ksize = 0; // Dims to be contracted

   if (transA == CblasNoTrans)
   {
      Msize = std::accumulate(extentA.begin(), extentA.begin()+M, 1ul, std::multiplies<size_type>());
      Ksize = std::accumulate(extentA.begin()+M, extentA.end(),   1ul, std::multiplies<size_type>());

      if (transB == CblasNoTrans)
         assert(std::equal(extentA.begin()+M, extentA.end(), extentB.begin()));
      else
         assert(std::equal(extentA.begin()+M, extentA.end(), extentB.begin()));
   }
   else
   {
      Msize = std::accumulate(extentA.begin()+K, extentA.end(),   1ul, std::multiplies<size_type>());
      Ksize = std::accumulate(extentA.begin(), extentA.begin()+K, 1ul, std::multiplies<size_type>());

      for (size_type i = 0; i < M; ++i) extentC[i] = extentA[K+i];

      if (transB == CblasNoTrans)
         assert(std::equal(extentA.begin()+M, extentA.end(), extentB.begin()+N));
      else
         assert(std::equal(extentA.begin()+M, extentA.end(), extentB.begin()+N));
   }

   if (transB == CblasNoTrans)
   {
      Nsize = std::accumulate(extentB.begin()+K, extentB.end(),   1ul, std::multiplies<size_type>());
   }
   else
   {
      Nsize = std::accumulate(extentB.begin(), extentB.begin()+N, 1ul, std::multiplies<size_type>());
   }

   if (c.empty()) {     // C empty -> compute extentC
     extentC = btas::array_adaptor<decltype(extentC)>::construct(M+N);
     if (transA == CblasNoTrans)
       for (size_type i = 0; i < M; ++i) extentC[i] = extentA[i];
     else
       for (size_type i = 0; i < M; ++i) extentC[i] = extentA[K+i];
     if (transB == CblasNoTrans)
       for (size_type i = 0; i < N; ++i) extentC[M+i] = extentB[K+i];
     else
       for (size_type i = 0; i < N; ++i) extentC[M+i] = extentB[i];
   }
   else { // C not empty -> validate extentC
     if (transA == CblasNoTrans)
       assert(std::equal(extentA.begin(), extentA.begin()+M, extentC.begin()));
     else
       assert(std::equal(extentA.begin()+K, extentA.end(), extentC.begin()));
     if (transB == CblasNoTrans)
       assert(std::equal(extentB.begin()+K, extentB.end(), extentC.begin()+M));
     else
       assert(std::equal(extentB.begin(), extentB.begin()+N, extentC.begin()+M));
   }

// // check iterator is "random-access" iterator
// static_assert(std::is_same<typename std::iterator_traits<typename _TensorA::iterator>::iterator_category, std::random_access_iterator_tag>::value, "gemm: _TensorA::iterator must be random-access iterator");
// static_assert(std::is_same<typename std::iterator_traits<typename _TensorB::iterator>::iterator_category, std::random_access_iterator_tag>::value, "gemm: _TensorB::iterator must be random-access iterator");
// static_assert(std::is_same<typename std::iterator_traits<typename _TensorC::iterator>::iterator_category, std::random_access_iterator_tag>::value, "gemm: _TensorC::iterator must be random-access iterator");

   typedef typename std::iterator_traits<typename _TensorA::iterator>::value_type value_type;

   // resize / scale
   if (c.empty())
   {
      c.resize(extentC);
      NumericType<value_type>::fill(c.begin(), c.end(), NumericType<value_type>::zero());
   }
   else
   {
      assert(std::equal(extentC.begin(), extentC.end(), extent(c).begin()));
      NumericType<value_type>::scal(c.begin(), c.end(), beta);
   }

   // Call GEMM depending on value type
   gemm_impl<std::is_convertible<_T, value_type>::value> call(transA, transB, Msize, Nsize, Ksize, alpha, a.begin(), b.begin(), beta, c.begin());
}

} // namespace btas

#endif // __BTAS_GEMM_IMPL_H
