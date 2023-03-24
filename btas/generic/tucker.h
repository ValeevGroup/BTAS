#ifndef BTAS_TUCKER_DECOMP_H
#define BTAS_TUCKER_DECOMP_H

#include <btas/generic/contract.h>
#include <btas/generic/core_contract.h>
#include <btas/generic/flatten.h>
#include <btas/generic/lapack_extensions.h>

#include <cstdlib>

namespace btas {

  /// Computes the tucker compression of an order-N tensor A.
  /// <a href=http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7516088> See
  /// reference. </a>
  /// First computes the tucker factor matrices for each mode of @a A then, if @a compute_core :
  /// @a A is transformed into the core tensor representation using the @a transforms

  /// \param[in, out] A In: Order-N tensor to be decomposed.  Out: if @a compute_core The core
  /// tensor of the Tucker decomposition else @a A \param[in] epsilon_svd The threshold
  /// truncation value for the Truncated Tucker-SVD decomposition
  ///  \param[in, out] transforms In: An empty vector.  Out: The Tucker factor matrices.
  /// \param[in] compute_core A bool which indicates if the tensor \c A should be transformed
  /// into the Tucker core matrices using the computed Tucker factor matrices stored in
  /// \c transforms.
  template<typename Tensor>
  void make_tucker_factors(Tensor& A, double epsilon_svd,
                           std::vector<Tensor> &transforms, bool compute_core = false){
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using dtype = typename Tensor::numeric_type;
    auto ndim = A.rank();
    transforms.clear();
    transforms.reserve(ndim);

    double norm2 = dot(A,A);
    auto threshold = epsilon_svd * epsilon_svd * norm2 / ndim;
    std::vector<size_t> left_modes, right_modes, final;
    final.push_back(0); final.emplace_back(ndim+1);
    left_modes.reserve(ndim); right_modes.reserve(ndim);
    for(ind_t i = 1; i <= ndim; ++i){
      left_modes.emplace_back(i);
      right_modes.emplace_back(i);
    }

    auto ptr_left = left_modes.begin(), ptr_right = right_modes.begin();
    for(ind_t i = 0; i < ndim; ++i, ++ptr_left, ++ptr_right){
      // Compute A * A to make tucker computation easier (this turns from SVD into an eigenvalue
      // decomposition, i.e. HOSVD)
      size_t temp = *ptr_left;
      *ptr_left = 0;
      *ptr_right = ndim + 1;
      Tensor AAt;
      contract(1.0, A, left_modes, A, right_modes, 0.0, AAt, final);
      *ptr_left = temp;
      *ptr_right = temp;

      // compute the eigenvalue decomposition of each mode of A
      ind_t r = AAt.extent(0);
      Tensor lambda(r); lambda.fill(0.0);
      auto info = hereig(blas::Layout::ColMajor, lapack::Job::Vec, lapack::Uplo::Lower, r, AAt.data(), r, lambda.data());
      if (info) BTAS_EXCEPTION("Error in computing the tucker SVD");

      // Find how many significant vectors are in this transformation
      ind_t rank = 0,  zero = 0;
      for(auto & eig : lambda){
        if(eig < threshold) ++rank;
      }

      // Truncate the column space of the unitary factor matrix.
      ind_t kept_evals = r - rank;
      if(kept_evals == 0) BTAS_EXCEPTION("Tucker decomposition failed. Tucker transformation rank = 0");
      lambda = Tensor(kept_evals, r);
      auto lower_bound = {rank, zero};
      auto upper_bound = {r, r};
      auto view = btas::make_view(AAt.range().slice(lower_bound, upper_bound), AAt.storage());
      std::copy(view.begin(), view.end(), lambda.begin());

      // Push the factor matrix back as a transformation.
      transforms.emplace_back(lambda);
    }

    if(compute_core){
      transform_tucker(true, A, transforms);
    }
  }

  /// Function much like `make_tucker_factors` however, after constructing
  /// nth factor of @a A, the nth mode of @a A is transformed into the
  /// core tensor space before the tucker factor of the (n+1)th mode is computed.
  /// \param[in, out] A In: Order-N tensor to be decomposed.  Out: The core
  /// tensor of the Tucker decomposition \param[in] epsilon_svd The threshold
  /// truncation value for the Truncated Tucker-SVD decomposition
  ///  \param[in, out] transforms In: An empty vector.  Out: The Tucker factor matrices.
  template<typename Tensor>
  void sequential_tucker(Tensor& A, double epsilon_svd, std::vector<Tensor> &transforms){
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using T = typename Tensor::numeric_type;
    using RT = real_type_t<T>;
    using RTensor = rebind_tensor_t<Tensor, RT>;
    auto ndim = A.rank();
    T one {1.0};
    T zero {0.0};
    transforms.clear();
    transforms.reserve(ndim);

    double norm2 = std::abs(dot(A,A));
    auto threshold = epsilon_svd * epsilon_svd * norm2 / ndim;
    std::vector<size_t> left_modes, right_modes, final, core;
    final.push_back(0); final.emplace_back(ndim);
    left_modes.reserve(ndim); right_modes.reserve(ndim);
    core.reserve(ndim);
    for(ind_t i = 0; i < ndim; ++i){
      left_modes.emplace_back(i);
      right_modes.emplace_back(i);
      core.emplace_back(i + 1);
    }
    *(right_modes.data()) = ndim;
    *(core.data() + ndim - 1) = 0;
    //auto ptr_left = left_modes.begin(), ptr_right = right_modes.begin();
    for(ind_t i = 0; i < ndim; ++i){
      // Compute A * A to make tucker computation easier (this turns from SVD into an eigenvalue
      // decomposition, i.e. HOSVD)
      // Because of later algorithm, mode of interest is always the 0th mode of the tensor
      Tensor AAt;
      contract(one , A, left_modes, A.conj(), right_modes, zero, AAt, final);

      // compute the eigenvalue decomposition of each mode of A
      ind_t r = AAt.extent(0);
      RTensor lambda(r); lambda.fill(0.0);
      auto info = hereig(blas::Layout::ColMajor, lapack::Job::Vec, lapack::Uplo::Lower, r, AAt.data(), r, lambda.data());
      if (info) BTAS_EXCEPTION("Error in computing the tucker SVD");

      // Find how many significant vectors are in this transformation
      ind_t rank = 0,  zero_ind = 0;
      for(auto & eig : lambda){
        if(eig < threshold) ++rank;
      }

      // Truncate the column space of the unitary factor matrix.
      ind_t kept_evals = r - rank;
      if(kept_evals == 0) BTAS_EXCEPTION("Tucker decomposition failed. Tucker transformation rank = 0");
      Tensor lambda_ (kept_evals, r);
      auto lower_bound = {rank, zero_ind};
      auto upper_bound = {r, r};
      auto view = btas::make_view(AAt.range().slice(lower_bound, upper_bound), AAt.storage());
      std::copy(view.begin(), view.end(), lambda_.begin());

      // Push the factor matrix back as a transformation.
      transforms.emplace_back(lambda_);

      // Now use lambda to move reference tensor to the core tensor space
      AAt = Tensor();
      contract(one, A, right_modes, lambda_.conj(), final, zero , AAt, core);
      A = AAt;
    }

  }

  /// A function to take an exact tensor to the Tucker core tensor or
  /// the Tucker core tensor to an approximation of the exact tensor.
  /// \param[in] to_core: if \c to_core tensor \c A will be taken from the exact representation to
  /// the Tucker core else \c A will be taken from the Tucker core representation to the exact representation
  /// \param[in, out] A In : depending on \c to_core an exact tensor or a Tucker core tensor. Out :
  /// a transformed tensor which represents either the Tucker core or exact tensor.
  /// \param[in] transforms the complete set of Tucker factor matrices. Note this does
  /// not include the core tensor.
  template<typename Tensor>
  void transform_tucker(bool to_core, Tensor & A, std::vector<Tensor> transforms){
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

    auto ndim = A.rank();

    std::vector<size_t> left_modes, right_modes, final;
    final.push_back(0); final.emplace_back(ndim+1);
    left_modes.reserve(ndim); right_modes.reserve(ndim);
    for(size_t i = 1; i <= ndim; ++i){
      left_modes.emplace_back(i);
      right_modes.emplace_back(i);
    }

    if(!to_core) {
      // as a reference, this properly flips the original tensor back to the original
      // subspace.
      auto ptr_tran = transforms.begin();

      for (size_t i = 0; i < ndim; ++i, ++ptr_tran) {
        right_modes.emplace_back(ndim + 1);
        right_modes.erase(right_modes.begin());
        left_modes[0] = 0;
        Tensor temp;

        contract(1.0, *ptr_tran, final, A, left_modes, 0.0, temp, right_modes);

        A = temp;
        right_modes[ndim - 1] = i + 1;
        left_modes = right_modes;
      }
    }
    else {
      ord_t size = A.size();
      right_modes = left_modes;
      auto ptr_tran = transforms.begin();
      // contracts the first mode and then tranposes it to the back of the tensor.
      // works like the s^N algebra does N rotations and then finished
      for(size_t i = 0; i < ndim; ++i, ++ptr_tran){
        right_modes.erase(right_modes.begin());
        right_modes.emplace_back(0);
        left_modes[0] = ndim + 1;
        Tensor temp;
        contract(1.0, *ptr_tran, final, A, left_modes, 0.0, temp, right_modes);

        A = temp;
        right_modes[ndim - 1] = i + 1;
        left_modes = right_modes;
      }
    }
  }

  /// Computes the tucker compression of an order-N tensor A.
  /// <a href=http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7516088> See
  /// reference. </a>

  /// \param[in, out] A In: Order-N tensor to be decomposed.  Out: The core
  /// tensor of the Tucker decomposition \param[in] epsilon_svd The threshold
  /// truncation value for the Truncated Tucker-SVD decomposition.
  /// \param[in, out] transforms In: An empty vector.  Out: The Tucker factor matrices.

  template <typename Tensor>
  [[deprecated]] void tucker_compression(Tensor &A, double epsilon_svd, std::vector<Tensor> &transforms) {
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