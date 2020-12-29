#ifndef BTAS_TUCKER_DECOMP_H
#define BTAS_TUCKER_DECOMP_H

#include <btas/generic/core_contract.h>
#include <btas/generic/flatten.h>
#include <btas/generic/contract.h>
#include <btas/generic/lapack_extensions.hpp>

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
  void tucker_compression(Tensor &A, double epsilon_svd, std::vector<Tensor> &transforms) {
    using ind_t = typename Tensor::range_type::index_type::value_type;
    auto ndim = A.rank();

    double norm2 = dot(A, A);
    auto threshold = epsilon_svd * epsilon_svd * norm2 / ndim;
    std::vector<size_t> first, second, final;
    for (size_t i = 0; i < ndim; ++i) {
      first.push_back(i);
      second.push_back(i);
    }
    final.push_back(0); final.push_back(ndim);
    auto ptr_second = second.begin(), ptr_final = final.begin();
    for (size_t i = 0; i < ndim; ++i, ++ptr_second) {
      Tensor S;

      *(ptr_second) = ndim;
      *(ptr_final) = i;
      contract(1.0, A, first, A, second, 0.0, S, final);
      *(ptr_second) = i;

      ind_t R = S.extent(0);
      Tensor lambda(R, 1);

      // Calculate the left singular vector of the flattened tensor
      // which is equivalent to the eigenvector of Flat \times Flat^T
      auto info = hereig(blas::Layout::RowMajor, lapack::Job::Vec, lapack::Uplo::Lower, R, S.data(), R, lambda.data());
      if (info) BTAS_EXCEPTION("Error in computing the tucker SVD");

      // Find the truncation rank based on the threshold.
      ind_t rank = 0;
      for (auto &eigvals : lambda) {
        if (eigvals < threshold) rank++;
      }

      // Truncate the column space of the unitary factor matrix.
      auto kept_evecs = R - rank;
      ind_t zero = 0;
      lambda = Tensor(R, kept_evecs);
      auto lower_bound = {zero, rank};
      auto upper_bound = {R, R};
      auto view = btas::make_view(S.range().slice(lower_bound, upper_bound), S.storage());
      std::copy(view.begin(), view.end(), lambda.begin());

      // Push the factor matrix back as a transformation.
      transforms.push_back(lambda);
    }

    // Make the second (the transformation modes)
    // order 2 and temp order N
    {
      auto temp = final;
      final = second;
      second = temp;
    }
    ptr_second = second.begin();
    ptr_final = final.begin();
    for (size_t i = 0; i < ndim; ++i, ++ptr_final) {
      auto &lambda = transforms[i];
#ifdef BTAS_HAS_INTEL_MKL
      // Contract the factor matrix with the reference tensor, A.
      core_contract(A, lambda, i);
#else
      Tensor rotated;
      // This multiplies by the transpose so later all I need to do is multiply by non-transpose
      second[0] = i; second[1] = ndim;
      *(ptr_final) = ndim;
      btas::contract(1.0, lambda, second, A, first, 0.0, rotated, final);
      *(ptr_final) = i;
      A = rotated;
#endif  // BTAS_HAS_INTEL_MKL
    }
  }
} // namespace btas
#endif // BTAS_TUCKER_DECOMP_H