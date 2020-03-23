#ifndef BTAS_TUCKER_DECOMP_H
#define BTAS_TUCKER_DECOMP_H

#include <btas/generic/core_contract.h>
#include <btas/generic/flatten.h>
#include <btas/generic/contract.h>

#include <cstdlib>

namespace btas {
/// Computes the tucker compression of an order-N tensor A.
/// <a href=http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7516088> See
/// reference. </a>

/// \param[in, out] A In: Order-N tensor to be decomposed.  Out: The core
/// tensor of the Tucker decomposition \param[in] epsilon_svd The threshold
/// truncation value for the Truncated Tucker-SVD decomposition \param[in, out]
/// transforms In: An empty vector.  Out: The Tucker factor matrices.

template <typename Tensor>
void tucker_compression(Tensor &A, double epsilon_svd,
                        std::vector<Tensor> &transforms) {
  double norm2 = dot(A, A);
  //norm2 *= norm2;
  unsigned int ndim = A.rank();
  std::vector<unsigned int> A_modes;
  for (unsigned int i = 0; i < ndim; ++i) {
    A_modes.push_back(i);
  }

  for (unsigned int i = 0; i < ndim; i++) {
    // Determine the threshold epsilon_SVD.
    auto flat = flatten(A, i);
    auto threshold = epsilon_svd * epsilon_svd * norm2 / ndim;

    std::uint64_t R = flat.extent(0);
    Tensor S(R, R), lambda(R, 1);

    // Contract A_n^T A_n to reduce the dimension of the SVD object to I_n X I_n
    gemm(CblasNoTrans, CblasTrans, 1.0, flat, flat, 0.0, S);

    // Calculate SVD of smaller object.
#ifdef BTAS_HAS_LAPACKE
    auto info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'L', R, S.data(), R,
                              lambda.data());
    if (info)
    BTAS_EXCEPTION("Error in computing the tucker SVD");
#else
    BTAS_EXCEPTION("Tucker decomposition requires LAPACKE");
#endif

    // Find the truncation rank based on the threshold.
    std::uint64_t rank = 0;
    for (auto &eigvals : lambda) {
      if (eigvals < threshold)
        rank++;
    }

    // Truncate the column space of the unitary factor matrix.
    auto kept_evecs = R - rank;
    lambda = Tensor(R, kept_evecs);
    auto lower_bound = {0, rank};
    auto upper_bound = {R, R};
    auto view =
        btas::make_view(S.range().slice(lower_bound, upper_bound), S.storage());
    std::copy(view.begin(), view.end(), lambda.begin());

    // Push the factor matrix back as a transformation.
    transforms.push_back(lambda);
  }

  for (unsigned int i = 0; i < ndim; ++i) {
    auto &lambda = transforms[i];
    std::uint64_t kept_evecs = lambda.extent(1);
#ifdef BTAS_HAS_INTEL_MKL
    // Contract the factor matrix with the reference tensor, A.
    core_contract(A, lambda, i);
#else
    std::vector<int> contract_modes;
    contract_modes.push_back(i);
    contract_modes.push_back(ndim);
    std::vector<int> final_modes;
    std::vector<int> final_dims;
    for (unsigned int j = 0; j < ndim; ++j) {
      if (j == i) {
        final_modes.push_back(ndim);
        final_dims.push_back(kept_evecs);
      } else {
        final_modes.push_back(j);
        final_dims.push_back(A.extent(j));
      }
    }
    btas::Range final_range(final_dims);
    Tensor final(final_range);
    btas::contract(1.0, A, A_modes, lambda, contract_modes, 0.0, final, final_modes);
    A = final;
#endif //BTAS_HAS_INTEL_MKL
  }
}
} // namespace btas
#endif // BTAS_TUCKER_DECOMP_H
