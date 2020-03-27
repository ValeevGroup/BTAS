#ifndef BTAS_KRP_H
#define BTAS_KRP_H

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
  using ind_t = typename Tensor::range_type::index_type::value_type;
  using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

  // Make sure the tensors are matrices
  if (A.rank() != 2 || B.rank() != 2) BTAS_EXCEPTION("A.rank() > 2 || B.rank() > 2, Matrices required");

  // Resize the product
  AB.resize(
          Range{Range1{A.extent(0) * B.extent(0)}, Range1{A.extent(1)}});

  // Calculate Khatri-Rao product by multiplying rows of A by rows of B.
  ind_t A_row = A.extent(0);
  ind_t B_row = B.extent(0);
  ind_t KRP_dim = A.extent(1);
  ord_t i_times_krp = 0, i_times_brow_krp = 0;
  for (ind_t i = 0; i < A_row; ++i, i_times_krp += KRP_dim) {
    const auto *A_ptr = A.data() + i_times_krp;
    ord_t j_times_KRP = 0;
    for (ind_t j = 0; j < B_row; ++j, j_times_KRP += KRP_dim) {
      const auto *B_ptr = B.data() + j_times_KRP;
      auto *AB_ptr = AB.data() + i_times_brow_krp + j_times_KRP;
      for (ind_t k = 0; k < KRP_dim; ++k) {
        *(AB_ptr + k) = *(A_ptr + k) * *(B_ptr + k);
      }
    }
    i_times_brow_krp += j_times_KRP;
  }
}

} // namespace btas

#endif // BTAS_KRP_H
