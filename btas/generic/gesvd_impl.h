#ifndef __BTAS_GESVD_IMPL_H
#define __BTAS_GESVD_IMPL_H 1

#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>

#include <btas/tensor_traits.h>
#include <btas/types.h>

#include <btas/generic/numeric_type.h>
#include <btas/generic/tensor_iterator_wrapper.h>

namespace btas {


template<bool _Finalize> struct gesvd_impl { };

template<> struct gesvd_impl<true>
{
   /// GESVD implementation
   template<class _IteratorA, class _IteratorS, class _IteratorU, class _IteratorVt>
   static void call (
      const CBLAS_ORDER& order,
      const char& jobu,
      const char& jobvt,
      const unsigned long& Msize,
      const unsigned long& Nsize,
            _IteratorA itrA,
      const unsigned long& LDA,
            _IteratorS itrS,
            _IteratorU itrU,
      const unsigned long& LDU,
            _IteratorVt itrVt,
      const unsigned long& LDVt)
   {
      assert(false); // gesvd_impl<true> for a generic iterator type has not yet been implemented.
   }

#ifdef BTAS_HAS_CBLAS

   static void call (
      const CBLAS_ORDER& order,
      const char& jobu,
      const char& jobvt,
      const unsigned long& Msize,
      const unsigned long& Nsize,
            float* itrA,
      const unsigned long& LDA,
            float* itrS,
            float* itrU,
      const unsigned long& LDU,
            float* itrVt,
      const unsigned long& LDVt)
   {
      unsigned long Ksize = (Msize < Nsize) ? Msize : Nsize;
      float* superb = new float[Ksize-1];
      LAPACKE_sgesvd(order, jobu, jobvt, Msize, Nsize, itrA, LDA, itrS, itrU, LDU, itrVt, LDVt, superb);
      delete [] superb;
   }

   static void call (
      const CBLAS_ORDER& order,
      const char& jobu,
      const char& jobvt,
      const unsigned long& Msize,
      const unsigned long& Nsize,
            double* itrA,
      const unsigned long& LDA,
            double* itrS,
            double* itrU,
      const unsigned long& LDU,
            double* itrVt,
      const unsigned long& LDVt)
   {
      unsigned long Ksize = (Msize < Nsize) ? Msize : Nsize;
      double* superb = new double[Ksize-1];
      LAPACKE_dgesvd(order, jobu, jobvt, Msize, Nsize, itrA, LDA, itrS, itrU, LDU, itrVt, LDVt, superb);
      delete [] superb;
   }

   static void call (
      const CBLAS_ORDER& order,
      const char& jobu,
      const char& jobvt,
      const unsigned long& Msize,
      const unsigned long& Nsize,
            std::complex<float>* itrA,
      const unsigned long& LDA,
            float* itrS,
            std::complex<float>* itrU,
      const unsigned long& LDU,
            std::complex<float>* itrVt,
      const unsigned long& LDVt)
   {
      unsigned long Ksize = (Msize < Nsize) ? Msize : Nsize;
      float* superb = new float[Ksize-1];
      LAPACKE_cgesvd(order, jobu, jobvt, Msize, Nsize, itrA, LDA, itrS, itrU, LDU, itrVt, LDVt, superb);
      delete [] superb;
   }

   static void call (
      const CBLAS_ORDER& order,
      const char& jobu,
      const char& jobvt,
      const unsigned long& Msize,
      const unsigned long& Nsize,
            std::complex<double>* itrA,
      const unsigned long& LDA,
            double* itrS,
            std::complex<double>* itrU,
      const unsigned long& LDU,
            std::complex<double>* itrVt,
      const unsigned long& LDVt)
   {
      unsigned long Ksize = (Msize < Nsize) ? Msize : Nsize;
      double* superb = new double[Ksize-1];
      LAPACKE_zgesvd(order, jobu, jobvt, Msize, Nsize, itrA, LDA, itrS, itrU, LDU, itrVt, LDVt, superb);
      delete [] superb;
   }

#endif // BTAS_HAS_CBLAS

};

template<> struct gesvd_impl<false>
{
   /// GESVD implementation
   template<class _IteratorA, class _IteratorS, class _IteratorU, class _IteratorVt>
   static void call (
      const CBLAS_ORDER& order,
      const char& jobu,
      const char& jobvt,
      const unsigned long& Msize,
      const unsigned long& Nsize,
            _IteratorA itrA,
      const unsigned long& LDA,
            _IteratorS itrS,
            _IteratorU itrU,
      const unsigned long& LDU,
            _IteratorVt itrVt,
      const unsigned long& LDVt)
   {
      assert(false); // gesvd_impl<false> for a generic iterator type has not yet been implemented.
   }

};

//  ================================================================================================

/// Generic implementation of BLAS GESVD in terms of C++ iterator
template<class _IteratorA, class _IteratorS, class _IteratorU, class _IteratorVt>
void gesvd (
   const CBLAS_ORDER& order,
   const char& jobu,
   const char& jobvt,
   const unsigned long& Msize,
   const unsigned long& Nsize,
         _IteratorA itrA,
   const unsigned long& LDA,
         _IteratorS itrS,
         _IteratorU itrU,
   const unsigned long& LDU,
         _IteratorVt itrVt,
   const unsigned long& LDVt)
{
   typedef std::iterator_traits<_IteratorA> __traits_A;
   typedef std::iterator_traits<_IteratorS> __traits_S;
   typedef std::iterator_traits<_IteratorU> __traits_U;
   typedef std::iterator_traits<_IteratorVt> __traits_Vt;

   typedef typename __traits_A::value_type value_type;

   static_assert(std::is_same<value_type, typename __traits_U::value_type>::value, "value type of U must be the same as that of A");
   static_assert(std::is_same<value_type, typename __traits_Vt::value_type>::value, "value type of Vt must be the same as that of A");

   static_assert(std::is_same<typename __traits_A::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator A must be a random access iterator");

   static_assert(std::is_same<typename __traits_S::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator S must be a random access iterator");

   static_assert(std::is_same<typename __traits_U::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator U must be a random access iterator");

   static_assert(std::is_same<typename __traits_Vt::iterator_category, std::random_access_iterator_tag>::value,
                 "iterator Vt must be a random access iterator");

   typename __traits_A::pointer A = &(*itrA);
   typename __traits_S::pointer S = &(*itrS);
   typename __traits_U::pointer U = &(*itrU);
   typename __traits_Vt::pointer Vt = &(*itrVt);
   gesvd_impl<true>::call(order, jobu, jobvt, Msize, Nsize, A, LDA, S, U, LDU, Vt, LDVt);
}

//  ================================================================================================

/// Generic interface of BLAS-GESVD
/// \param order storage order of tensor in matrix view (CblasRowMajor, CblasColMajor)
/// \param transA transpose directive for tensor A (CblasNoTrans, CblasTrans, CblasConjTrans)
/// \param alpha scalar value to be multiplied to A * X
/// \param A input tensor
/// \param X input tensor
/// \param beta scalar value to be multiplied to Y
/// \param Y output tensor which can be empty tensor but needs to have rank info (= size of shape).
/// Iterator is assumed to be consecutive (or, random_access_iterator) , thus e.g. iterator to map doesn't work.
template<
   class _TensorA, class _VectorS, class _TensorU, class _TensorVt,
   class = typename std::enable_if<
      is_boxtensor<_TensorA>::value &
      is_boxtensor<_TensorU>::value &
      is_boxtensor<_TensorVt>::value
   >::type
>
void gesvd (
   const char& jobu,
   const char& jobvt,
         _TensorA& A,
         _VectorS& S,
         _TensorU& U,
         _TensorVt& Vt)
{
    static_assert(boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorU>::value &&
                  boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorVt>::value,
                  "btas::gesvd does not support mixed storage order");
    const CBLAS_ORDER order = boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorA>::row_major ?
                              CblasRowMajor : CblasColMajor;

   assert(!A.empty());

   const size_type rankA = rank(A);
   const size_type rankU = rank(U);
   const size_type rankVt = rank(Vt);

   assert(rankA == (rankU+rankVt-2));

   // get shapes
   const typename _TensorA::range_type::extent_type& extentA = extent(A);
         typename _VectorS::range_type::extent_type extentS = extent(S); // if S is empty, this gives { 0,...,0 }
         typename _TensorU::range_type::extent_type extentU = extent(U); // if U is empty, this gives { 0,...,0 }
         typename _TensorVt::range_type::extent_type extentVt = extent(Vt); // if Vt is empty, this gives { 0,...,0 }

   size_type Msize = 0; // Rows count of Y
   size_type Nsize = 0; // Cols count of Y

   size_type LDA = 0; // Leading dims of A
   size_type LDU = 0; // Leading dims of U
   size_type LDVt = 0; // Leading dims of Vt

   Msize = std::accumulate(std::begin(extentA), std::begin(extentA)+rankU-1, 1ul, std::multiplies<size_type>());
   Nsize = std::accumulate(std::begin(extentA)+rankU-1, std::end(extentA),   1ul, std::multiplies<size_type>());

   size_type Ksize = std::min(Msize,Nsize);
   size_type Ucols = (jobu == 'A') ? Msize : Ksize;
   size_type Vtrows = (jobvt == 'A') ? Nsize : Ksize;

   extentS[0] = Ksize;

   for (size_type i = 0; i < rankU-1; ++i) extentU[i] = extentA[i];
   extentU[rankU-1] = Ucols;

   extentVt[0] = Vtrows;
   for (size_type i = 1; i < rankVt; ++i) extentVt[i] = extentA[i+rankU-2];

   if(order == CblasRowMajor)
   {
      LDA = Nsize;
      LDU = Ucols;
      LDVt = Nsize;
   }
   else
   {
      LDA = Msize;
      LDU = Msize;
      LDVt = Vtrows;
   }

   S.resize(extentS);
   U.resize(extentU);
   Vt.resize(extentVt);

   auto itrA = std::begin(A);
   auto itrS = std::begin(S);
   auto itrU = std::begin(U);
   auto itrVt = std::begin(Vt);

   gesvd (order, jobu, jobvt, Msize, Nsize, itrA, LDA, itrS, itrU, LDU, itrVt, LDVt);
}

} // namespace btas

#endif // __BTAS_GESVD_IMPL_H
