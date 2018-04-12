#ifndef BTAS_KRP_H
#define BTAS_KRP_H

#include <btas/btas.h>

namespace btas {

/// The khatri-rao product is an outer product of column vectors of \param A
/// and \param B, the product is then ordered to make a super column in a new matrix
/// The dimension of this product is \f[ A(N, M) \cdot B(K, M) = AB(N*K , M)\f ]

/// \param[in] A Matrix of size (N, M)
/// \param[in] B Matrix of size (K, M)
/// \param[in, out] AB In: Matrix of any size. Out: Matrix of size (N*K,
/// M)

template <class Tensor>
void khatri_rao_product(const Tensor &A, const Tensor &B, Tensor &AB) {
  // Make sure the tensors are matrices
  if (A.rank() != 2 || B.rank() != 2)
    BTAS_EXCEPTION("A.rank() > 2 || B.rank() > 2, Matrices required");
  
  // Resize the product
  AB.resize(
      Range{Range1{A.extent(0) * B.extent(0)}, Range1{A.extent(1)}});
  
  // Calculate Khatri-Rao product by multiplying rows of A by rows of B.
  auto A_row = A.extent(0);
  auto B_row = B.extent(0);
  auto KRP_dim = A.extent(1);
  for (auto i = 0; i < A_row; ++i) {
    const auto *A_ptr = A.data() + i * KRP_dim;
    for (auto j = 0; j < B_row; ++j) {
      const auto * B_ptr = B.data() + j * KRP_dim;
      auto * AB_ptr = AB.data() + i * B_row * KRP_dim + j * KRP_dim;
      for (auto k = 0; k < KRP_dim; ++k) {
        //AB(i * B.extent(0) + j, k) = A(i, k) * B(j, k);
        *(AB_ptr + k) = *(A_ptr + k) * *(B_ptr + k);
      }
    }
  }
}

} // namespace btas

#endif // BTAS_KRP_H
