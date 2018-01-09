#ifndef BTAS_KRP_H
#define BTAS_KRP_H

#include <btas/btas.h>

namespace btas {

/// The khatri-rao product is an outer product of column vectors of A
/// and B, the product is then ordered to make a super column in a new matrix
/// The dimension of this product is  A(NXM) (.) B(KXM) = AB_product(N*K X M)

/// \param[in] A Matrix of size (N, M)
/// \param[in] B Matrix of size (K, M)
/// \param[in, out] AB_product In: Matrix of any size, Out: Matrix of size (N*K,
/// M)

template <class Tensor>
void khatri_rao_product(const Tensor &A, const Tensor &B, Tensor &AB_product) {
  if (A.rank() > 2 || B.rank() > 2)
    BTAS_EXCEPTION("A.rank() > 2 || B.rank() > 2, Matrices required");
  
  AB_product.resize(
      Range{Range1{A.extent(0) * B.extent(0)}, Range1{A.extent(1)}});
  
  for (auto i = 0; i < A.extent(0); ++i)
    for (auto j = 0; j < B.extent(0); ++j)
      for (auto k = 0; k < A.extent(1); ++k)
        AB_product(i * B.extent(0) + j, k) = A(i, k) * B(j, k);
}

} // namespace btas

#endif // BTAS_KRP_H
