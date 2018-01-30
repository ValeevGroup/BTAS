#ifndef BTAS_TUCKER_DECOMP_H
#define BTAS_TUCKER_DECOMP_H

#ifdef _HAS_INTEL_MKL

#include "core_contract.h"
#include "flatten.h"
#include <btas/btas.h>

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
  double norm2 = norm(A);
  norm2 *= norm2;
  auto ndim = A.rank();

  for (int i = 0; i < ndim; i++) {
    // Determine the threshold epsilon_SVD.
    auto flat = flatten(A, i);
    auto threshold = epsilon_svd * epsilon_svd * norm2 / ndim;

    int R = flat.extent(0);
    Tensor S(R, R), lambda(R, 1);

    // Contract A_n^T A_n to reduce the dimension of the SVD object to I_n X I_n
    gemm(CblasNoTrans, CblasTrans, 1.0, flat, flat, 0.0, S);

    // Calculate SVD of smaller object.
    auto info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'L', R, S.data(), R,
                              lambda.data());
    if (info)
      BTAS_EXCEPTION("Error in computing the tucker SVD");

    // Find the truncation rank based on the threshold.
    int rank = 0;
    for (auto &eigvals : lambda) {
      if (eigvals < threshold)
        rank++;
    }

    // Truncate the column space of the unitary factor matrix.
    lambda = Tensor(R, R - rank);
    auto lower_bound = {0, rank};
    auto upper_bound = {R, R};
    auto view =
        btas::make_view(S.range().slice(lower_bound, upper_bound), S.storage());
    std::copy(view.begin(), view.end(), lambda.begin());

    // Push the factor matrix back as a transformation.
    transforms.push_back(lambda);

    // Contract the factor matrix with the reference tensor, A.
    core_contract(A, lambda, i);
  }
}

/// calculates the 2-norm of a matrix Mat
/// \param[in] Mat The matrix who's 2-norm is caclulated

template <typename Tensor> double norm(const Tensor &Mat) {
  return sqrt(dot(Mat, Mat));
}
} // namespace btas
#endif //_HAS_INTEL_MKL
#endif // BTAS_TUCKER_DECOMP_H
