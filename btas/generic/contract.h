#ifndef __BTAS_CONTRACT_H
#define __BTAS_CONTRACT_H

#include <algorithm>
#include <memory>

#include <btas/types.h>
#include <btas/tensor_traits.h>

#include <btas/util/resize.h>

#include <btas/generic/scal_impl.h>
#include <btas/generic/gemv_impl.h>
#include <btas/generic/ger_impl.h>
#include <btas/generic/gemm_impl.h>
#include <btas/generic/permute.h>

namespace btas {

/// contract tensors; for example, Cijk = \sum_{m,p} Aimp * Bmjpk
///
/// Synopsis:
/// enum {j,k,l,m,n,o};
///
/// contract(alpha,A,{m,o,k,n},B,{l,k,j},C,beta,{l,n,m,o,j});
///
///       o       j           o j
///       |       |           | |
///   m - A - k - B   =   m -  C
///       |       |           | |
///       n       l           n l
///
/// NOTE: in case of TArray, this performs many unuse instances of gemv and gemm depend on tensor rank
///
template<
   typename _T,
   class _TensorA, class _TensorB, class _TensorC,
   class _AnnotationA, class _AnnotationB, class _AnnotationC,
   class = typename std::enable_if<
      is_boxtensor<_TensorA>::value &
      is_boxtensor<_TensorB>::value &
      is_boxtensor<_TensorC>::value &
      is_container<_AnnotationA>::value &
      is_container<_AnnotationB>::value &
      is_container<_AnnotationC>::value
   >::type
>
void contract(
   const _T& alpha,
   const _TensorA& A, const _AnnotationA& aA,
   const _TensorB& B, const _AnnotationB& aB,
   const _T& beta,
         _TensorC& C, const _AnnotationC& aC)
{
   // check index A
   auto __sort_indexA = _AnnotationA{aA};
   std::sort(__sort_indexA.begin(), __sort_indexA.end());
   assert(std::unique(__sort_indexA.begin(), __sort_indexA.end()) == __sort_indexA.end());

   // check index B
   auto __sort_indexB = _AnnotationB{aB};
   std::sort(__sort_indexB.begin(), __sort_indexB.end());
   assert(std::unique(__sort_indexB.begin(), __sort_indexB.end()) == __sort_indexB.end());

   // check index C
   auto __sort_indexC = _AnnotationC{aC};
   std::sort(__sort_indexC.begin(), __sort_indexC.end());
   assert(std::unique(__sort_indexC.begin(), __sort_indexC.end()) == __sort_indexC.end());

   typedef btas::varray<size_t> Permutation;

   // permute index A
   Permutation __permute_indexA;
   resize(__permute_indexA, aA.size());

   // permute index B
   Permutation __permute_indexB;
   resize(__permute_indexB, aB.size());

   // permute index C
   Permutation __permute_indexC;
   resize(__permute_indexC, aC.size());

   size_type m = 0;
   size_type n = 0;
   size_type k = 0;

   // row index
   for(auto itrA = aA.begin(); itrA != aA.end(); ++itrA)
   {
      if(!std::binary_search(__sort_indexB.begin(), __sort_indexB.end(), *itrA))
      {
         __permute_indexA[m] = *itrA;
         __permute_indexC[m] = *itrA;
         ++m;
      }
   }
   // index to be contracted
   for(auto itrA = aA.begin(); itrA != aA.end(); ++itrA)
   {
      if( std::binary_search(__sort_indexB.begin(), __sort_indexB.end(), *itrA))
      {
         __permute_indexA[m+k] = *itrA;
         __permute_indexB[k]   = *itrA;
         ++k;
      }
   }
   // column index
   for(auto itrB = aB.begin(); itrB != aB.end(); ++itrB)
   {
      if(!std::binary_search(__sort_indexA.begin(), __sort_indexA.end(), *itrB))
      {
         __permute_indexB[k+n] = *itrB;
         __permute_indexC[m+n] = *itrB;
         ++n;
      }
   }

   // check result index C
   Permutation __sort_permute_indexC(__permute_indexC);
   std::sort(__sort_permute_indexC.begin(), __sort_permute_indexC.end());
   assert(std::equal(__sort_permute_indexC.begin(), __sort_permute_indexC.end(), __sort_indexC.begin()));

   // permute A
   std::shared_ptr<const _TensorA> __refA;
   if(!std::equal(aA.begin(), aA.end(), __permute_indexA.begin()))
   {
      __refA = std::make_shared<const _TensorA>();
      permute(A, aA, const_cast<_TensorA&>(*__refA), __permute_indexA);
   }
   else
   {
      __refA.reset(&A, btas::nulldeleter());
   }

   // permute B
   std::shared_ptr<const _TensorB> __refB;
   if(!std::equal(aB.begin(), aB.end(), __permute_indexB.begin()))
   {
      __refB = std::make_shared<const _TensorB>();
      permute(B, aB, const_cast<_TensorB&>(*__refB), __permute_indexB);
   }
   else
   {
      __refB.reset(&B, btas::nulldeleter());
   }

   bool __C_to_permute = false;

   // to set rank of C
   if(C.empty())
   {
      Permutation __zero_shape;
      resize(__zero_shape, m+n);
      std::fill(__zero_shape.begin(), __zero_shape.end(), 0);
      C.resize(__zero_shape);
   }

   // permute C
   std::shared_ptr<_TensorC> __refC;
   if(!std::equal(aC.begin(), aC.end(), __permute_indexC.begin()))
   {
      __refC = std::make_shared<_TensorC>();
      permute(C, aC, *__refC, __permute_indexC);
      __C_to_permute = true;
   }
   else
   {
      __refC.reset(&C, btas::nulldeleter());
   }

   // call BLAS functions
   if     (A.rank() == k && B.rank() == k)
   {
      assert(false); // dot should be called instead
   }
   else if(k == 0)
   {
      scal(beta, *__refC);
      ger (alpha, *__refA, *__refB, *__refC);
   }
   else if(A.rank() == k)
   {
      gemv(CblasTrans,   alpha, *__refB, *__refA, beta, *__refC);
   }
   else if(B.rank() == k)
   {
      gemv(CblasNoTrans, alpha, *__refA, *__refB, beta, *__refC);
   }
   else
   {
      gemm(CblasNoTrans, CblasNoTrans, alpha, *__refA, *__refB, beta, *__refC);
   }

   // permute back
   if(__C_to_permute)
   {
      permute(*__refC, __permute_indexC, C, aC);
   }
}

template<
   typename _T,
   class _TensorA, class _TensorB, class _TensorC,
   typename _UA, typename _UB, typename _UC,
   class = typename std::enable_if<
      is_tensor<_TensorA>::value &
      is_tensor<_TensorB>::value &
      is_tensor<_TensorC>::value &
      std::is_same<typename _TensorA::value_type, typename _TensorB::value_type>::value &
      std::is_same<typename _TensorA::value_type, typename _TensorC::value_type>::value
   >::type
>
void contract(
   const _T& alpha,
   const _TensorA& A, std::initializer_list<_UA> aA,
   const _TensorB& B, std::initializer_list<_UB> aB,
   const _T& beta,
         _TensorC& C, std::initializer_list<_UC> aC)
{
    contract(alpha,
             A, btas::varray<_UA>(aA),
             B, btas::varray<_UB>(aB),
             beta,
             C, btas::varray<_UC>(aC)
            );
}

} //namespace btas

#endif
