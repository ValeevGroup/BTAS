#ifndef __BTAS_CONTRACT_H
#define __BTAS_CONTRACT_H

namespace btas
{

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
template<typename _T, class _TensorA, class _TensorB, class _TensorC>
void contract(
      const _T& alpha,
      const _TensorA& A, const typename _TensorA::shape_type& Ashape,
      const _TensorB& B, const typename _TensorB::shape_type& Bshape,
      const _T& beta,
            _TensorC& C, const typename _TensorC::shape_type& Cshape)
{
}

} //namespace btas

#endif
