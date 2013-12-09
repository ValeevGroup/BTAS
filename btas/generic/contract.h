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
   class = typename std::enable_if<
      is_tensor<_TensorA>::value &
      is_tensor<_TensorB>::value &
      is_tensor<_TensorC>::value &
      std::is_same<typename _TensorA::value_type, typename _TensorB::value_type>::value &
      std::is_same<typename _TensorA::value_type, typename _TensorC::value_type>::value
   >::type
>
void contract(
   const CBLAS_ORDER& order,
   const _T& alpha,
   const _TensorA& A, const typename _TensorA::shape_type& indexA,
   const _TensorB& B, const typename _TensorB::shape_type& indexB,
   const _T& beta,
         _TensorC& C, const typename _TensorC::shape_type& indexC)
{
   // check index A
   typename _TensorA::shape_type __sort_indexA(indexA);
   std::sort(__sort_indexA.begin(), __sort_indexA.end());
   assert(std::unique(__sort_indexA.begin(), __sort_indexA.end()) == __sort_indexA.end());

   // check index B
   typename _TensorB::shape_type __sort_indexB(indexB);
   std::sort(__sort_indexB.begin(), __sort_indexB.end());
   assert(std::unique(__sort_indexB.begin(), __sort_indexB.end()) == __sort_indexB.end());

   // check index C
   typename _TensorC::shape_type __sort_indexC(indexC);
   std::sort(__sort_indexC.begin(), __sort_indexC.end());
   assert(std::unique(__sort_indexC.begin(), __sort_indexC.end()) == __sort_indexC.end());

   // permute index A
   typename _TensorA::shape_type __permute_indexA;
   resize(__permute_indexA, indexA.size());

   // permute index B
   typename _TensorB::shape_type __permute_indexB;
   resize(__permute_indexB, indexB.size());

   // permute index C
   typename _TensorC::shape_type __permute_indexC;
   resize(__permute_indexC, indexC.size());

   size_type m = 0;
   size_type n = 0;
   size_type k = 0;

   // row index
   for(auto itrA = indexA.begin(); itrA != indexA.end(); ++itrA)
   {
      if(!std::binary_search(__sort_indexB.begin(), __sort_indexB.end(), *itrA))
      {
         __permute_indexA[m] = *itrA;
         __permute_indexC[m] = *itrA;
         ++m;
      }
   }
   // index to be contracted
   for(auto itrA = indexA.begin(); itrA != indexA.end(); ++itrA)
   {
      if( std::binary_search(__sort_indexB.begin(), __sort_indexB.end(), *itrA))
      {
         __permute_indexA[m+k] = *itrA;
         __permute_indexB[k]   = *itrA;
         ++k;
      }
   }
   // column index
   for(auto itrB = indexB.begin(); itrB != indexB.end(); ++itrB)
   {
      if(!std::binary_search(__sort_indexA.begin(), __sort_indexA.end(), *itrB))
      {
         __permute_indexB[k+n] = *itrB;
         __permute_indexC[m+n] = *itrB;
         ++n;
      }
   }

   // check result index C
   typename _TensorC::shape_type __sort_permute_indexC(__permute_indexC);
   std::sort(__sort_permute_indexC.begin(), __sort_permute_indexC.end());
   assert(std::equal(__sort_permute_indexC.begin(), __sort_permute_indexC.end(), __sort_indexC.begin()));

   // permute A
   std::shared_ptr<const _TensorA> __refA;
   if(!std::equal(indexA.begin(), indexA.end(), __permute_indexA.begin()))
   {
      __refA = std::make_shared<const _TensorA>();
      permute(A, indexA, const_cast<_TensorA&>(*__refA), __permute_indexA);
   }
   else
   {
      __refA.reset(&A, btas::nulldeleter());
   }

   // permute B
   std::shared_ptr<const _TensorB> __refB;
   if(!std::equal(indexB.begin(), indexB.end(), __permute_indexB.begin()))
   {
      __refB = std::make_shared<const _TensorB>();
      permute(B, indexB, const_cast<_TensorB&>(*__refB), __permute_indexB);
   }
   else
   {
      __refB.reset(&B, btas::nulldeleter());
   }

   bool __C_to_permute = false;

   // to set rank of C
   if(C.empty())
   {
      typename _TensorC::shape_type __zero_shape;
      resize(__zero_shape, m+n);
      std::fill(__zero_shape.begin(), __zero_shape.end(), 0);
      C.resize(__zero_shape);
   }

   // permute C
   std::shared_ptr<_TensorC> __refC;
   if(!std::equal(indexC.begin(), indexC.end(), __permute_indexC.begin()))
   {
      __refC = std::make_shared<_TensorC>();
      permute(C, indexC, *__refC, __permute_indexC);
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
      ger (order, alpha, *__refA, *__refB, *__refC);
   }
   else if(A.rank() == k)
   {
      gemv(order, CblasTrans,   alpha, *__refB, *__refA, beta, *__refC);
   }
   else if(B.rank() == k)
   {
      gemv(order, CblasNoTrans, alpha, *__refA, *__refB, beta, *__refC);
   }
   else
   {
      gemm(order, CblasNoTrans, CblasNoTrans, alpha, *__refA, *__refB, beta, *__refC);
   }

   // permute back
   if(__C_to_permute)
   {
      permute(*__refC, __permute_indexC, C, indexC);
   }
}

} //namespace btas

#endif
