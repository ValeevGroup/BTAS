#ifndef __BTAS_CONTRACT_H
#define __BTAS_CONTRACT_H

namespace btas {

template<bool _DoContract> struct contract_impl { };

/*! contract tensors; for example, Cijk = \sum_{m,p} Aimp * Bmjpk
 *
 * Synopsis:
 * enum {j,k,l,m,n,o};
 *
 * void
 * contract(alpha,A,shape(m,o,k,n),B,shape(l,k,j),C,beta,shape(l,n,m,o,j));
 *
 *       o       j           o j
 *       |       |           | |
 *   m - A - k - B   =   m -  C
 *       |       |           | |
 *       n       l           n l
 *
 */
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
   const _T& alpha,
   const _TensorA& A, const typename _TensorA::shape_type& shapeA,
   const _TensorB& B, const typename _TensorB::shape_type& shapeB,
   const _T& beta,
         _TensorC& C, const typename _TensorC::shape_type& shapeC)
{
   typedef std::iterator_traits<typename _TensorA::iterator> __traits;
   contract_impl<std::is_convertible<_T, typename __traits::value_type>::value> call(alpha, A, shapeA, B, shapeB, beta, C, shapeC);
}

template<> struct contract_impl<true>
{
   template<typename _T, class _TensorA, class _TensorB, class _TensorC>
   contract_impl(
      const _T& alpha,
      const _TensorA& A, const typename _TensorA::shape_type& shapeA,
      const _TensorB& B, const typename _TensorB::shape_type& shapeB,
      const _T& beta,
            _TensorC& C, const typename _TensorC::shape_type& shapeC)
   {
   }
};

template<> struct contract_impl<false>
{
   template<typename _T, class _TensorA, class _TensorB, class _TensorC>
   contract_impl(
      const _T& alpha,
      const _TensorA& A, const typename _TensorA::shape_type& shapeA,
      const _TensorB& B, const typename _TensorB::shape_type& shapeB,
      const _T& beta,
            _TensorC& C, const typename _TensorC::shape_type& shapeC)
   {
   }
};

} //namespace btas

#endif
