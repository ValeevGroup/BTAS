#ifndef __BTAS_GEMM_IMPL_H
#define __BTAS_GEMM_IMPL_H 1

#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>

#include <btas/tensor.h>
#include <btas/tensor_traits.h>
#include <btas/generic/numeric_type.h>
#include <btas/types.h>
#include <btas/array_adaptor.h>
#include <btas/generic/tensor_iterator_wrapper.h>

#include <btas/generic/scal_impl.h>

namespace btas {

template<bool _Finalize> struct gemm_impl { };




template<> struct gemm_impl<true>
{
   template<typename _T, class _IteratorA, class _IteratorB, class _IteratorC>
   static void call_impl (
      const blas::Layout& order,
      const blas::Op& transA,
      const blas::Op& transB,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const unsigned long& Ksize,
      const _T& alpha,
            _IteratorA itrA,
      const unsigned long& LDA,
            _IteratorB itrB,
      const unsigned long& LDB,
      const _T& beta,
            _IteratorC itrC,
      const unsigned long& LDC,
      generic_impl_tag)
   {
      // For column-major order, recall this as C^T = B^T * A^T in row-major order
      if (order == blas::Layout::ColMajor)
      {
         gemm_impl<true>::call(blas::Layout::RowMajor, transB, transA, Nsize, Msize, Ksize, alpha, itrB, LDB, itrA, LDA, beta, itrC, LDC);

         return;
      }

      if (beta == NumericType<_T>::zero())
      {
         std::fill_n(itrC, Msize*Nsize, NumericType<typename std::iterator_traits<_IteratorC>::value_type>::zero());
      }
      else if (beta != NumericType<_T>::one())
      {
         scal (Msize*Nsize, beta, itrC, 1);
      }

      // A:NoTrans / B:NoTrans
      if (transA == blas::Op::NoTrans && transB == blas::Op::NoTrans)
      {
         auto itrB_save = itrB;
         auto itrC_save = itrC;
         for (size_type i = 0; i < Msize; ++i)
         {
            itrB = itrB_save;
            for (size_type k = 0; k < Ksize; ++k, ++itrA)
            {
               itrC = itrC_save;
               for (size_type j = 0; j < Nsize; ++j, ++itrB, ++itrC)
               {
                  (*itrC) += alpha * (*itrA) * (*itrB);
               }
            }
            itrC_save += Nsize;
         }
      }
      // A:NoTrans / B:Trans
      else if (transA == blas::Op::NoTrans && transB == blas::Op::Trans)
      {
         auto itrA_save = itrA;
         auto itrB_save = itrB;
         for (size_type i = 0; i < Msize; ++i)
         {
            itrB = itrB_save;
            for (size_type j = 0; j < Nsize; ++j, ++itrC)
            {
               itrA = itrA_save;
               for (size_type k = 0; k < Ksize; ++k, ++itrA, ++itrB)
               {
                  (*itrC) += alpha * (*itrA) * (*itrB);
               }
            }
            itrA_save += Ksize;
         }
      }
      // A:NoTrans / B:ConjTrans
      else if (transA == blas::Op::NoTrans && transB == blas::Op::ConjTrans)
      {
         auto itrA_save = itrA;
         auto itrB_save = itrB;
         for (size_type i = 0; i < Msize; ++i)
         {
            itrB = itrB_save;
            for (size_type j = 0; j < Nsize; ++j, ++itrC)
            {
               itrA = itrA_save;
               for (size_type k = 0; k < Ksize; ++k, ++itrA, ++itrB)
               {
                  (*itrC) += alpha * (*itrA) * impl::conj(*itrB);
               }
            }
            itrA_save += Ksize;
         }
      }
      // A:Trans / B:NoTrans
      else if (transA == blas::Op::Trans && transB == blas::Op::NoTrans)
      {
         auto itrB_save = itrB;
         auto itrC_save = itrC;
         for (size_type k = 0; k < Ksize; ++k)
         {
            itrC = itrC_save;
            for (size_type i = 0; i < Msize; ++i, ++itrA)
            {
               itrB = itrB_save;
               for (size_type j = 0; j < Nsize; ++j, ++itrB, ++itrC)
               {
                  (*itrC) += alpha * (*itrA) * (*itrB);
               }
            }
            itrB_save += Nsize;
         }
      }
      // A:ConjTrans / B:NoTrans
      else if (transA == blas::Op::ConjTrans && transB == blas::Op::NoTrans)
      {
         auto itrB_save = itrB;
         auto itrC_save = itrC;
         for (size_type k = 0; k < Ksize; ++k)
         {
            itrC = itrC_save;
            for (size_type i = 0; i < Msize; ++i, ++itrA)
            {
               itrB = itrB_save;
               for (size_type j = 0; j < Nsize; ++j, ++itrB, ++itrC)
               {
                  (*itrC) += alpha * impl::conj(*itrA) * (*itrB);
               }
            }
            itrB_save += Nsize;
         }
      }
      // A:Trans / B:Trans
      else if (transA == blas::Op::Trans && transB == blas::Op::Trans)
      {
         auto itrA_save = itrA;
         auto itrC_save = itrC;
         for (size_type j = 0; j < Nsize; ++j, ++itrC_save)
         {
            itrA = itrA_save;
            for (size_type k = 0; k < Ksize; ++k, ++itrB)
            {
               itrC = itrC_save;
               for (size_type i = 0; i < Msize; ++i, ++itrA, itrC += Nsize)
               {
                  (*itrC) += alpha * (*itrA) * (*itrB);
               }
            }
         }
      }
      // A:Trans / B:ConjTrans
      else if (transA == blas::Op::Trans && transB == blas::Op::ConjTrans)
      {
         auto itrA_save = itrA;
         auto itrC_save = itrC;
         for (size_type j = 0; j < Nsize; ++j, ++itrC_save)
         {
            itrA = itrA_save;
            for (size_type k = 0; k < Ksize; ++k, ++itrB)
            {
               itrC = itrC_save;
               for (size_type i = 0; i < Msize; ++i, ++itrA, itrC += Nsize)
               {
                  (*itrC) += alpha * (*itrA) * impl::conj(*itrB);
               }
            }
         }
      }
      // A:ConjTrans / B:Trans
      else if (transA == blas::Op::ConjTrans && transB == blas::Op::Trans)
      {
         auto itrA_save = itrA;
         auto itrC_save = itrC;
         for (size_type j = 0; j < Nsize; ++j, ++itrC_save)
         {
            itrA = itrA_save;
            for (size_type k = 0; k < Ksize; ++k, ++itrB)
            {
               itrC = itrC_save;
               for (size_type i = 0; i < Msize; ++i, ++itrA, itrC += Nsize)
               {
                  (*itrC) += alpha * impl::conj(*itrA) * (*itrB);
               }
            }
         }
      }
      // A:ConjTrans / B:ConjTrans
      else if (transA == blas::Op::ConjTrans && transB == blas::Op::ConjTrans)
      {
         auto itrA_save = itrA;
         auto itrC_save = itrC;
         for (size_type j = 0; j < Nsize; ++j, ++itrC_save)
         {
            itrA = itrA_save;
            for (size_type k = 0; k < Ksize; ++k, ++itrB)
            {
               itrC = itrC_save;
               for (size_type i = 0; i < Msize; ++i, ++itrA, itrC += Nsize)
               {
                  (*itrC) += alpha * impl::conj(*itrA) * impl::conj(*itrB);
               }
            }
         }
      }
      else {
        assert(false);
      }

   }


#ifdef BTAS_HAS_BLAS_LAPACK
   template<typename _T, class _IteratorA, class _IteratorB, class _IteratorC>
   static void call_impl (
      const blas::Layout& order,
      const blas::Op& transA,
      const blas::Op& transB,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const unsigned long& Ksize,
      const _T& alpha,
            _IteratorA itrA,
      const unsigned long& LDA,
            _IteratorB itrB,
      const unsigned long& LDB,
      const _T& beta,
            _IteratorC itrC,
      const unsigned long& LDC,
      blas_lapack_impl_tag)
   {

      using a_traits = std::iterator_traits<_IteratorA>;
      using b_traits = std::iterator_traits<_IteratorB>;
      using c_traits = std::iterator_traits<_IteratorC>;

      using a_value_type = typename a_traits::value_type;
      using b_value_type = typename b_traits::value_type;
      using c_value_type = typename c_traits::value_type;

      using a_ptr_type = const a_value_type*;
      using b_ptr_type = const b_value_type*;
      using c_ptr_type =       c_value_type*;

      blas::gemm( order, transA, transB, Msize, Nsize, Ksize, alpha,
                  static_cast<a_ptr_type>(&(*itrA)), LDA,
                  static_cast<b_ptr_type>(&(*itrB)), LDB,
                  beta,
                  static_cast<c_ptr_type>(&(*itrC)), LDC );
    }
#endif

   template<typename _T, class _IteratorA, class _IteratorB, class _IteratorC>
   static void call (
      const blas::Layout& order,
      const blas::Op& transA,
      const blas::Op& transB,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const unsigned long& Ksize,
      const _T& alpha,
            _IteratorA itrA,
      const unsigned long& LDA,
            _IteratorB itrB,
      const unsigned long& LDB,
      const _T& beta,
            _IteratorC itrC,
      const unsigned long& LDC)
   {

     call_impl( order, transA, transB, Msize, Nsize, Ksize, alpha, itrA, LDA,
                itrB, LDB, beta, itrC, LDC, 
                blas_lapack_impl_t<_IteratorA,_IteratorB,_IteratorC>() );

   }

};






#if 1
template<> struct gemm_impl<false>
{
   template<typename _T, class _IteratorA, class _IteratorB, class _IteratorC>
   static void call (
      const blas::Layout& order,
      const blas::Op& transA,
      const blas::Op& transB,
      const unsigned long& Msize,
      const unsigned long& Nsize,
      const unsigned long& Ksize,
      const _T& alpha,
            _IteratorA itrA,
      const unsigned long& LDA,
            _IteratorB itrB,
      const unsigned long& LDB,
      const _T& beta,
            _IteratorC itrC,
      const unsigned long& LDC)
   {
      // currently, column-major order has not yet been supported at this level
      assert(order == blas::Layout::RowMajor);

      if (beta == NumericType<_T>::zero())
      {
         std::fill_n(itrC, Msize*Nsize, NumericType<typename std::iterator_traits<_IteratorC>::value_type>::zero());
      }
      else if (beta != NumericType<_T>::one())
      {
         scal (Msize*Nsize, beta, itrC, 1);
      }

      // A:NoTrans / B:NoTrans
      if (transA == blas::Op::NoTrans && transB == blas::Op::NoTrans)
      {
         auto itrB_save = itrB;
         auto itrC_save = itrC;
         for (size_type i = 0; i < Msize; ++i)
         {
            itrB = itrB_save;
            for (size_type k = 0; k < Ksize; ++k, ++itrA)
            {
               itrC = itrC_save;
               for (size_type j = 0; j < Nsize; ++j, ++itrB, ++itrC)
               {
                   gemm(transA, transB, alpha, *itrA, *itrB, beta, *itrC);
               }
            }
            itrC_save += Nsize;
         }
      }
      // A:NoTrans / B:Trans
      else if (transA == blas::Op::NoTrans && transB != blas::Op::NoTrans)
      {
         auto itrA_save = itrA;
         auto itrB_save = itrB;
         for (size_type i = 0; i < Nsize; ++i)
         {
            itrB = itrB_save;
            for (size_type j = 0; j < Msize; ++j, ++itrC)
            {
               itrA = itrA_save;
               for (size_type k = 0; k < Ksize; ++k, ++itrA, ++itrB)
               {
                  gemm(transA, transB, alpha, *itrA, *itrB, beta, *itrC);
               }
            }
            itrA_save += Ksize;
         }
      }
      // A:Trans / B:NoTrans
      else if (transA != blas::Op::NoTrans && transB == blas::Op::NoTrans)
      {
         auto itrB_save = itrB;
         auto itrC_save = itrC;
         for (size_type k = 0; k < Ksize; ++k)
         {
            itrC = itrC_save;
            for (size_type i = 0; i < Msize; ++i, ++itrA)
            {
               itrB = itrB_save;
               for (size_type j = 0; j < Nsize; ++j, ++itrB, ++itrC)
               {
                  gemm(transA, transB, alpha, *itrA, *itrB, beta, *itrC);
               }
            }
            itrB_save += Nsize;
         }
      }
      // A:Trans / B:Trans
      else if (transA != blas::Op::NoTrans && transB != blas::Op::NoTrans)
      {
         auto itrA_save = itrA;
         auto itrC_save = itrC;
         for (size_type j = 0; j < Nsize; ++j, ++itrC_save)
         {
            itrA = itrA_save;
            for (size_type k = 0; k < Ksize; ++k, ++itrB)
            {
               itrC = itrC_save;
               for (size_type i = 0; i < Msize; ++i, ++itrA, itrC += Nsize)
               {
                  gemm(transA, transB, alpha, *itrA, *itrB, beta, *itrC);
               }
            }
         }
      }
   }
};
#endif

//  ================================================================================================

/// Generic implementation of BLAS GEMM in terms of C++ iterator
template<typename _T, class _IteratorA, class _IteratorB, class _IteratorC>
void gemm (
   const blas::Layout& order,
   const blas::Op& transA,
   const blas::Op& transB,
   const unsigned long& Msize,
   const unsigned long& Nsize,
   const unsigned long& Ksize,
   const _T& alpha,
         _IteratorA itrA,
   const unsigned long& LDA,
         _IteratorB itrB,
   const unsigned long& LDB,
   const _T& beta,
         _IteratorC itrC,
   const unsigned long& LDC)
{
   typedef std::iterator_traits<_IteratorA> __traits_A;
   typedef std::iterator_traits<_IteratorB> __traits_B;
   typedef std::iterator_traits<_IteratorC> __traits_C;

   typedef typename __traits_A::value_type value_type;

   static_assert(std::is_same<value_type, typename __traits_B::value_type>::value, "value type of B must be the same as that of A");
   static_assert(std::is_same<value_type, typename __traits_C::value_type>::value, "value type of C must be the same as that of A");

   static_assert(std::is_same<typename __traits_A::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator A must be a random access iterator");

   static_assert(std::is_same<typename __traits_B::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator B must be a random access iterator");

   static_assert(std::is_same<typename __traits_C::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator C must be a random access iterator");

   typename __traits_A::pointer A = &(*itrA);
   typename __traits_B::pointer B = &(*itrB);
   typename __traits_C::pointer C = &(*itrC);
   gemm_impl<std::is_convertible<_T, value_type>::value>::call(order, transA, transB, Msize, Nsize, Ksize, alpha, A, LDA, B, LDB, beta, C, LDC);

}

//  ================================================================================================

/// Generic implementation of BLAS-GEMM
/// \param transA transpose directive for tensor \p a (blas::Op)
/// \param transB transpose directive for tensor \p b (blas::Op)
/// \param alpha scalar value to be multiplied to \param a * \param b
/// \param a input tensor
/// \param b input tensor
/// \param beta scalar value to be multiplied to \param c
/// \param c output tensor which can be empty tensor but needs to have rank info
template<
   typename _T,
   class _TensorA, class _TensorB, class _TensorC,
   class = typename std::enable_if<
      is_boxtensor<_TensorA>::value &
      is_boxtensor<_TensorB>::value &
      is_boxtensor<_TensorC>::value &
      std::is_same<typename _TensorA::value_type, typename _TensorB::value_type>::value &
      std::is_same<typename _TensorA::value_type, typename _TensorC::value_type>::value
   >::type
>
void gemm (
   const blas::Op& transA,
   const blas::Op& transB,
   const _T& alpha,
   const _TensorA& A,
   const _TensorB& B,
   const _T& beta,
         _TensorC& C)
{
   static_assert(boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorC>::value &&
                 boxtensor_storage_order<_TensorB>::value == boxtensor_storage_order<_TensorC>::value,
                 "btas::gemm does not support mixed storage order");
   static_assert(boxtensor_storage_order<_TensorC>::value != boxtensor_storage_order<_TensorC>::other,
                 "btas::gemm does not support non-major storage order");
   const blas::Layout order = boxtensor_storage_order<_TensorC>::value == boxtensor_storage_order<_TensorC>::row_major ?
                             blas::Layout::RowMajor : blas::Layout::ColMajor;

   typedef unsigned long size_type;

   //if (A.empty() || B.empty()) return;
   //assert (C.rank() != 0);

   if (A.empty() || B.empty())
   {
      scal(beta, C);
      return;
   }

   typedef typename _TensorA::value_type value_type;
   assert(not ((transA == blas::Op::ConjTrans || transB == blas::Op::ConjTrans) && std::is_fundamental<value_type>::value));


   // get contraction rank
   const size_type rankA = rank(A);
   const size_type rankB = rank(B);
   const size_type rankC = rank(C);

   const size_type K = (rankA+rankB-rankC)/2; assert((rankA+rankB-rankC) % 2 == 0);
   const size_type M = rankA-K;
   const size_type N = rankB-K;

   // get extents
   auto extentA = extent(A);
   auto extentB = extent(B);
   typename _TensorC::range_type::extent_type extentC = extent(C); // if C is empty, this gives { }, will need to allocate

   size_type Msize = 0; // Rows count of C
   size_type Nsize = 0; // Cols count of C
   size_type Ksize = 0; // Dims to be contracted

   size_type LDA = 0; // Leading dims of A
   size_type LDB = 0; // Leading dims of B
   size_type LDC = 0; // Leading dims of C

   Msize = (transA == blas::Op::NoTrans)
           ? std::accumulate(std::begin(extentA), std::begin(extentA)+M, 1ul, std::multiplies<size_type>())
           : std::accumulate(std::begin(extentA)+K, std::end(extentA),   1ul, std::multiplies<size_type>())
   ;
   Ksize = (transA == blas::Op::NoTrans)
           ? std::accumulate(std::begin(extentA)+M, std::end(extentA),   1ul, std::multiplies<size_type>())
           : std::accumulate(std::begin(extentA), std::begin(extentA)+K, 1ul, std::multiplies<size_type>())
   ;

   // check that contraction dimensions match
   auto Barea = range(B).area();
   {
     // weak check
     assert(Barea % Ksize == 0);

     // strong checks
     if (transA == blas::Op::NoTrans && transB == blas::Op::NoTrans)
       assert(std::equal(std::begin(extentA)+M, std::end(extentA), std::begin(extentB)));
     if (transA == blas::Op::NoTrans && transB != blas::Op::NoTrans)
       assert(std::equal(std::begin(extentA)+M, std::end(extentA), std::begin(extentB)+N));
     if (transA != blas::Op::NoTrans && transB == blas::Op::NoTrans)
       assert(std::equal(std::begin(extentA), std::begin(extentA)+K, std::begin(extentB)));
     if (transA != blas::Op::NoTrans && transB != blas::Op::NoTrans)
       assert(std::equal(std::begin(extentA), std::begin(extentA)+K, std::begin(extentB)+N));
   }

   Nsize = Barea / Ksize;

   if(order == blas::Layout::RowMajor) {
     if(transA == blas::Op::NoTrans) LDA = Ksize;
     else LDA = Msize;
     if(transB == blas::Op::NoTrans) LDB = Nsize;
     else LDB = Ksize;
     LDC = Nsize;
   }
   else {
     if(transA == blas::Op::NoTrans) LDA = Msize;
     else LDA = Ksize;
     if(transB == blas::Op::NoTrans) LDB = Ksize;
     else LDB = Msize;
     LDA = Msize;
     LDB = Ksize;
     LDC = Msize;
   }

   if (C.empty()) {     // C empty -> compute extentC
     extentC = btas::array_adaptor<decltype(extentC)>::construct(M+N);
     if (transA == blas::Op::NoTrans)
       for (size_type i = 0; i < M; ++i) extentC[i] = extentA[i];
     else
       for (size_type i = 0; i < M; ++i) extentC[i] = extentA[K+i];
     if (transB == blas::Op::NoTrans)
       for (size_type i = 0; i < N; ++i) extentC[M+i] = extentB[K+i];
     else
       for (size_type i = 0; i < N; ++i) extentC[M+i] = extentB[i];
   }
   else { // C not empty -> validate extentC
     if (transA == blas::Op::NoTrans)
       assert(std::equal(std::begin(extentA), std::begin(extentA)+M, std::begin(extentC)));
     else
       assert(std::equal(std::begin(extentA)+K, std::end(extentA), std::begin(extentC)));
     if (transB == blas::Op::NoTrans)
       assert(std::equal(std::begin(extentB)+K, std::end(extentB), std::begin(extentC)+M));
     else
       assert(std::equal(std::begin(extentB), std::begin(extentB)+N, std::begin(extentC)+M));
   }

   // resize / scale
   if (C.empty())
   {
      C.resize(extentC);
      NumericType<value_type>::fill(std::begin(C), std::end(C), NumericType<value_type>::zero());
   }
   else
   {
      assert(std::equal(std::begin(extentC), std::end(extentC), std::begin(extent(C))));
      if (beta == NumericType<_T>::zero())
        NumericType<value_type>::fill(std::begin(C), std::end(C), NumericType<value_type>::zero());
   }

   auto itrA = std::begin(A);
   auto itrB = std::begin(B);
   auto itrC = std::begin(C);

   gemm (order, transA, transB, Msize, Nsize, Ksize, alpha, itrA, LDA, itrB, LDB, beta, itrC, LDC);
}

} // namespace btas

#endif // __BTAS_GEMM_IMPL_H
