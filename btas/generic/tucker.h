#ifndef BTAS_TUCKER_DECOMP_H
#define BTAS_TUCKER_DECOMP_H

#ifdef _HAS_INTEL_MKL

#include "core_contract.h"
#include "flatten.h"
#include <btas/btas.h>

#include <cstdlib>

namespace btas {
/// Computes the tucker compression of a Nth order tensor A.
/// See http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7516088

/// \param[in, out] A In: Tucker decomposition is applied to A.  Out: The core
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
    Tensor S(R, R), s_real(R, 1), s_image(R, 1), U(R, R), Vl(1, 1);
    s_real.fill(0.0);
    s_image.fill(0.0);
    U.fill(0.0);
    Vl.fill(0.0);

    // Contract A_n^T A_n to reduce the dimension of the SVD object to I_n X
    // I_n.
    gemm(CblasNoTrans, CblasTrans, 1.0, flat, flat, 0.0, S);

    // Calculate SVD of smaller object.
    auto info =
        LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', R, S.data(), R, s_real.data(),
                      s_image.data(), Vl.data(), R, U.data(), R);
    if (info)
      return;

    // Find the truncation rank based on the threshold.
    int rank = 0;
    for (auto &eigvals : s_real)
      if (eigvals > threshold)
        rank++;
    s_real = Tensor(0);

    // Truncate the column space of the unitary factor matrix.
    auto lower_bound = {0, 0};
    auto upper_bound = {R, rank};
    auto view =
        btas::make_view(U.range().slice(lower_bound, upper_bound), U.storage());
    Vl.resize(Range{Range1{R}, Range1{rank}});
    std::copy(view.begin(), view.end(), Vl.begin());

    // Push the factor matrix back as a transformation.
    transforms.push_back(Vl);

    // Contract the factor matrix with the reference tensor, A.
    core_contract(A, Vl, i);
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
