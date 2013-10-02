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
template <typename T>
void
contract(T alpha, 
         const Tensor<T>& A, const shape& Ashape,
         const Tensor<T>& B, const shape Bshape,
         T beta,
         Tensor<T>& C,const shape& Cshape);

} //namespace btas

#endif
