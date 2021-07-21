#ifndef __BTAS_GESVD_IMPL_H
#define __BTAS_GESVD_IMPL_H 1

#include <algorithm>
#include <numeric>
#include <iterator>
#include <type_traits>

#include <btas/tensor.h>
#include <btas/tensor_traits.h>
#include <btas/types.h>
#include <btas/type_traits.h>

#include <btas/generic/blas_lapack_delegator.h>
#include <btas/generic/lapack_extensions.h>
#include <btas/generic/numeric_type.h>
#include <btas/generic/tensor_iterator_wrapper.h>

namespace btas {


template<bool _Finalize> struct gesvd_impl { };

template<> struct gesvd_impl<true>
{
   /// GESVD implementation
   template<class _IteratorA, class _IteratorS, class _IteratorU, class _IteratorVt>
   static void call_impl (
      const blas::Layout& order,
      lapack::Job jobu,
      lapack::Job jobvt,
      const unsigned long& Msize,
      const unsigned long& Nsize,
            _IteratorA itrA,
      const unsigned long& LDA,
            _IteratorS itrS,
            _IteratorU itrU,
      const unsigned long& LDU,
            _IteratorVt itrVt,
      const unsigned long& LDVt,
      generic_impl_tag)
   {
      BTAS_EXCEPTION("GESVD Does not have a Generic Implementation");
   }


#ifdef BTAS_HAS_BLAS_LAPACK
   template<class _IteratorA, class _IteratorS, class _IteratorU, class _IteratorVt>
   static void call_impl (
      const blas::Layout& order,
      lapack::Job jobu,
      lapack::Job jobvt,
      const unsigned long& Msize,
      const unsigned long& Nsize,
            _IteratorA itrA,
      const unsigned long& LDA,
            _IteratorS itrS,
            _IteratorU itrU,
      const unsigned long& LDU,
            _IteratorVt itrVt,
      const unsigned long& LDVt,
      blas_lapack_impl_tag)
   {

     using value_type = typename std::iterator_traits<_IteratorA>::value_type;
     using real_type  = real_type_t<value_type>;

     const bool needU     = jobu  != lapack::Job::NoVec;
     const bool needVt    = jobvt != lapack::Job::NoVec;
     const bool inplaceU  = jobu  == lapack::Job::OverwriteVec;
     const bool inplaceVt = jobvt == lapack::Job::OverwriteVec;

     if( inplaceU and inplaceVt ) 
       BTAS_EXCEPTION("SVD cannot return both vectors inplace");



     value_type dummy;
     value_type* A = static_cast<value_type*>(&(*itrA));
     value_type* U = (needU and not inplaceU) ?
                      static_cast<value_type*>(&(*itrU)) : &dummy;
     value_type* Vt = (needVt and not inplaceVt) ?
                      static_cast<value_type*>(&(*itrVt)) : &dummy;

     real_type* S = static_cast<real_type*> (&(*itrS));

     auto info = gesvd( order, jobu, jobvt, Msize, Nsize, A, LDA, S, U, LDU, Vt, LDVt );

 
     if( info ) BTAS_EXCEPTION("SVD Failed");     

     
   }
#endif

   template<class _IteratorA, class _IteratorS, class _IteratorU, class _IteratorVt>
   static void call (
      const blas::Layout& order,
      lapack::Job jobu,
      lapack::Job jobvt,
      const unsigned long& Msize,
      const unsigned long& Nsize,
            _IteratorA itrA,
      const unsigned long& LDA,
            _IteratorS itrS,
            _IteratorU itrU,
      const unsigned long& LDU,
            _IteratorVt itrVt,
      const unsigned long& LDVt )
   {

     call_impl( order, jobu, jobvt, Msize, Nsize, itrA, LDA, itrS, itrU, LDU,
                itrVt, LDVt, 
                blas_lapack_impl_t<_IteratorA,_IteratorS,_IteratorU,_IteratorVt>() );

   }


};

template<> struct gesvd_impl<false>
{
   /// GESVD implementation
   template<class _IteratorA, class _IteratorS, class _IteratorU, class _IteratorVt>
   static void call (
      const blas::Layout& order,
      lapack::Job jobu,
      lapack::Job jobvt,
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
   const blas::Layout& order,
   lapack::Job jobu,
   lapack::Job jobvt,
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
/// \param order storage order of tensor in matrix view (blas::Layout)
/// \param transA transpose directive for tensor \p A (blas::Op)
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
   lapack::Job jobu,
   lapack::Job jobvt,
         _TensorA& A,
         _VectorS& S,
         _TensorU& U,
         _TensorVt& Vt)
{
    static_assert(boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorU>::value &&
                  boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorVt>::value,
                  "btas::gesvd does not support mixed storage order");
    const blas::Layout order = boxtensor_storage_order<_TensorA>::value == boxtensor_storage_order<_TensorA>::row_major ?
                              blas::Layout::RowMajor : blas::Layout::ColMajor;

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
   size_type Ucols = (jobu == lapack::Job::AllVec) ? Msize : Ksize;
   size_type Vtrows = (jobvt == lapack::Job::AllVec) ? Nsize : Ksize;

   extentS[0] = Ksize;

   for (size_type i = 0; i < rankU-1; ++i) extentU[i] = extentA[i];
   extentU[rankU-1] = Ucols;

   extentVt[0] = Vtrows;
   for (size_type i = 1; i < rankVt; ++i) extentVt[i] = extentA[i+rankU-2];

   if(order == blas::Layout::RowMajor)
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
