#ifndef BTAS_TUCKER
#define BTAS_TUCKER
#ifdef _HAS_INTEL_MKL
#include "core_contract.h"
#include "flatten.h"
#include <btas/btas.h>
#include <stdlib.h>
namespace btas {
  /// Computes the tucker compression of a Nth order tensor A.
  /// See http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7516088
template <typename Tensor>
void tucker_compression(Tensor &A, double epsilon_svd,
                        std::vector<Tensor> &transforms) {
  double norm2 = norm(A);
  norm2 *= norm2;
  auto ndim = A.rank();

  for (int i = 0; i < ndim; i++) {
    auto flat = flatten(A, i);
    auto threshold = epsilon_svd * epsilon_svd * norm2 / ndim;
    int R = flat.extent(0);
    Tensor S(R, R), s_real(R, 1), s_image(R, 1), U(R, R), Vl(1, 1);
    s_real.fill(0.0);
    s_image.fill(0.0);
    U.fill(0.0);
    Vl.fill(0.0);

    gemm(CblasNoTrans, CblasTrans, 1.0, flat, flat, 0.0, S);
    auto info =
        LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', R, S.data(), R, s_real.data(),
                      s_image.data(), Vl.data(), R, U.data(), R);
    if (info)
      return;

    int rank = 0;
    Tensor Ipiv(R, R);
    Ipiv.fill(0.0);
    // Eigenvalues are unsorted
    // This algorithm sorts the eigenvalues and generates the piviting matrix
    // for the right eigenvector which reduces the column dimension to rank
    for (int j = 0; j < R; j++) {
      auto hold = -1e10;
      int swap = 0;
      for (int k = 0; k < R; k++) {
        if (s_real(k, 0) > hold) {
          hold = s_real(k, 0);
          swap = k;
        }
      }
      s_real(swap, 0) = -1e10;
      if (hold < epsilon_svd)
        break;
      Ipiv(j, swap) = 1.0;
      rank++;
    }
    s_real = Tensor(0);
    S.resize(Range{Range1{R}, Range1{R}});
    gemm(CblasNoTrans, CblasTrans, 1.0, U, Ipiv, 0.0, S);
    auto lower_bound = {0, 0};
    auto upper_bound = {R, rank};
    auto view =
        btas::make_view(S.range().slice(lower_bound, upper_bound), S.storage());
    U.resize(Range{Range1{R}, Range1{rank}});
    std::copy(view.begin(), view.end(), U.begin());
    transforms.push_back(U);
    core_contract(A, U, i);
  }
}

template <typename Tensor> double norm(const Tensor &Mat) {
  return sqrt(dot(Mat, Mat));
}
} // namespace btas
#endif //_HAS_INTEL_MKL
#endif // BTAS_TUCKER
