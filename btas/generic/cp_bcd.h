//
// Created by Karl Pierce on 4/29/23.
//

#ifndef BTAS_GENERIC_CP_BCD_H
#define BTAS_GENERIC_CP_BCD_H

#include <btas/generic/cp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace btas{
  /** \brief Computes the Canonical Product (CP) decomposition of an order-N
    tensor using block coordinate descent (BCD).

  This computes the CP decomposition of btas::Tensor objects with row
  major storage only with fixed (compile-time) and variable (run-time)
  ranks. Also provides Tucker and randomized Tucker-like compressions coupled
  with CP-BCD decomposition. Does not support strided ranges.

   \warning this code takes a non-const reference \c tensor_ref but does
   not modify the values. This is a result of API (reshape needs non-const tensor)

   Synopsis:
    \code
      // Constructors
      CP_BCD A(tensor)         // CP_BCD object with empty factor
                               // matrices and no symmetries
      CP_BCD A(tensor, symms)  // CP_ALS object with empty factor
                               // matrices and symmetries

      // Operations
      A.compute_rank(rank, converge_test)  // Computes the CP_BCD of tensor to
                                           // rank, rank build and HOSVD options

      A.compute_rank_random(rank, converge_test)  // Computes the CP_BCD of tensor to
                                                  // rank. Factor matrices built at rank
                                                  // with random numbers

      A.compute_error(converge_test, omega)  // Computes the CP_BCD of tensor to
                                             // 2-norm
                                             // error < omega.

      //See documentation for full range of options

      // Accessing Factor Matrices
      A.get_factor_matrices()  // Returns a vector of factor matrices, if
                               // they have been computed

      A.reconstruct()  // Returns the tensor computed using the
                       // CP factor matrices
    \endcode
                                         */

  template <typename Tensor, class ConvClass = NormCheck<Tensor> >
  class CP_BCD : public CP_ALS<Tensor, ConvClass> {
   public:
    using T = typename Tensor::value_type;
    using RT = real_type_t<T>;
    using RTensor = rebind_tensor_t<Tensor, RT>;

    using CP<Tensor, ConvClass>::A;
    using CP<Tensor, ConvClass>::ndim;
    using CP<Tensor, ConvClass>::symmetries;
    using typename CP<Tensor, ConvClass>::ind_t;
    using typename CP<Tensor, ConvClass>::ord_t;
    using CP_ALS<Tensor, ConvClass>::tensor_ref;
    using CP_ALS<Tensor, ConvClass>::size;

    /// Create a CP BCD object, child class of the CP object
    /// that stores the reference tensor.
    /// Reference tensor has no symmetries.
    /// \param[in] tensor the reference tensor to be decomposed.
    CP_BCD(Tensor &tensor, ind_t block_size = 1, size_t sweeps=1) : CP_ALS<Tensor, ConvClass>(tensor), blocksize(block_size), nsweeps(sweeps){
    }

    /// Create a CP BCD object, child class of the CP object
    /// that stores the reference tensor.
    /// Reference tensor has symmetries.
    /// Symmetries should be set such that the higher modes index
    /// are set equal to lower mode indices (a 4th order tensor,
    /// where the second & third modes are equal would have a
    /// symmetries of {0,1,1,3}
    /// \param[in] tensor the reference tensor to be decomposed.
    /// \param[in] symms the symmetries of the reference tensor.
    CP_BCD(Tensor &tensor, std::vector<size_t> &symms, ind_t block_size = 1, size_t sweeps=1)
        : CP_ALS<Tensor, ConvClass>(tensor, symms), blocksize(block_size), nsweeps(sweeps){
    }

    CP_BCD() = default;

    ~CP_BCD() = default;

   protected:
    Tensor gradient, block_ref;           // Stores gradient of BCD for fitting.
    ind_t blocksize;           // Size of block gradient
    std::vector<Tensor> blockfactors;
    std::vector<size_t> order;  // convenient to write order of tensors for reconstruct
    long z = 0;
    size_t nsweeps;

    /// computed the CP decomposition using ALS to minimize the loss function for fixed rank \p rank
    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit.
    /// \param[in] dir The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS. Default = 1e5.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated ||T_exact - T_approx|| = epsilon.
    /// \param[in] tcutALS
    /// How small difference in factor matrices must be to consider ALS of a
    /// single rank converged. Default = 0.1.
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in,out] fast_pI Whether the pseudo inverse be computed using a fast cholesky decomposition,
    ///       on return \c fast_pI will be true if use of Cholesky was successful
    virtual void ALS(ind_t rank, ConvClass &converge_test, bool dir, int max_als, bool calculate_epsilon, double &epsilon,
                     bool &fast_pI){
      for(auto i = 0; i < ndim; ++i){
        order.emplace_back(i);
      }

      block_ref = tensor_ref;

      // Number of full blocks with rank blocksize
      // plus one more block with rank rank % blocksize
      int n_full_blocks = int( rank / blocksize),
          last_blocksize = rank % blocksize;

      // copying all of A to block factors. Then we are going to take slices of A
      // that way we can leverage all of `CP_ALS` code without modification
      blockfactors = A;
      gradient = tensor_ref;
      // this stores the factors from rank 0 to #blocks * blocksize
      // Compute all the BCD of the full blocks
      auto matlab = true;
      auto one_over_tref = 1.0 / this->norm(tensor_ref);
      auto fit = 1.0, change = 0.0;
      bool compute_full_fit = false;
      for(auto s = 0; s < nsweeps; ++s){
        long block_step = 0;
        this->AtA = std::vector<Tensor>(ndim);
        for (long b = 0; b < n_full_blocks; ++b, block_step += blocksize) {
          BCD(block_step, block_step + blocksize, max_als, fast_pI, matlab, converge_test, s);
          // Test the system to see if converged. Doing the hard way first
          // To compute the accuracy fully compute CP approximation using blocks 0 through b.
          if(compute_full_fit) {
            std::vector<Tensor> current_grads(ndim);
            for (size_t i = 0; i < ndim; ++i) {
              auto &a_mat = blockfactors[i];
              auto lower = {z, z}, upper = {long(a_mat.extent(0)), block_step + blocksize};
              current_grads[i] = make_view(a_mat.range().slice(lower, upper), a_mat.storage());
            }

            auto temp = reconstruct(current_grads, order, blockfactors[ndim]);
            auto newfit = 1.0 - this->norm(temp - tensor_ref) * one_over_tref;
            change = abs(fit - newfit);
            fit = newfit;
            std::cout << block_step + blocksize << "\t";
            std::cout << fit << "\t" << change << std::endl;
          }
        }
        if(last_blocksize != 0) {
          this->AtA = std::vector<Tensor>(ndim);
          block_step = n_full_blocks * blocksize;
          BCD(block_step, block_step + last_blocksize, max_als, fast_pI, matlab, converge_test, s);
          if(compute_full_fit) {
            std::vector<Tensor> current_grads(ndim);
            for (size_t i = 0; i < ndim; ++i) {
              auto &a_mat = blockfactors[i];
              auto lower = {z, z}, upper = {long(a_mat.extent(0)), block_step + last_blocksize};
              current_grads[i] = make_view(a_mat.range().slice(lower, upper), a_mat.storage());
            }

            auto temp = reconstruct(current_grads, order, blockfactors[ndim]);
            auto newfit = 1.0 - this->norm(temp - tensor_ref) * one_over_tref;
            change = abs(fit - newfit);
            fit = newfit;
            std::cout << block_step + last_blocksize << "\t";
            std::cout << fit << "\t" << change << std::endl;
          }
        }
        epsilon = (compute_full_fit == false ? 
                        this->norm(tensor_ref - reconstruct(blockfactors, order, blockfactors[ndim]))
                        : 1.0 - fit);
      }
      A = blockfactors;
    }

    void copy_blocks(Tensor & to, const Tensor & from, ind_t block_start, ind_t block_end){
        auto tref_dim = to.extent(0);
        auto to_rank = to.extent(1), from_rank = from.extent(1);

        auto to_ptr = to.data();
        auto from_ptr = from.data();
        ind_t from_pos = 0;
        for (ind_t i = 0, skip = 0; i < tref_dim; ++i, skip += to_rank){
          for (auto b = block_start; b < block_end; ++b, ++from_pos){
            *(to_ptr + skip + b) = *(from_ptr + from_pos);
          }
        }
    }
    /// Computes an optimized factor matrix holding all others constant.
    /// No Khatri-Rao product computed, immediate contraction

    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    /// \param[in,out] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// return if computing the \c fast_pI was successful.
    /// \param[in, out] matlab If \c fast_pI = true then try to solve VA = B instead of taking pseudoinverse
    /// in the same manner that matlab would compute the inverse. If this fails, variable will be manually
    /// set to false and SVD will be used.
    /// return if \c matlab was successful
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit. test to see if the ALS is converged

    void BCD(ind_t block_start, ind_t block_end, size_t max_als,
             bool &fast_pI, bool &matlab, ConvClass &converge_test, size_t sweep, double lambda = 0.0){
      // Take the b-th block of the factor matrix also take the
      // find the partial grammian for the blocked factors
      auto cur_block_size = block_end - block_start;
      for (size_t i = 0; i < ndim; ++i){
        auto & a_mat = A[i];
        auto lower = {z, block_start}, upper = {long(a_mat.extent(0)), block_end};
        a_mat = make_view(blockfactors[i].range().slice(lower, upper), blockfactors[i].storage());
        contract(this->one, a_mat, {1, 2}, a_mat.conj(), {1, 3}, this->zero, this->AtA[i], {2, 3});
      }
      A[ndim] = Tensor(cur_block_size);
      A[ndim].fill(0.0);
      if(sweep != 0){
        size_t c = 0;
        auto lam_full_ptr = blockfactors[ndim].data(), new_lam_ptr = A[ndim].data();
        for(auto b = block_start; b < block_end; ++b, ++c)
          *(new_lam_ptr + c) = *(lam_full_ptr + b);
        
        gradient += reconstruct(A, order, A[ndim]);
      }
      // Do the ALS loop
      //CP_ALS::ALS(cur_block_size, converge_test, dir, max_als, calculate_epsilon, epsilon, fast_pI);
      // First do it manually, so we know its right
      // Compute ALS of the bth block against the gradient
      // computed by subtracting the previous blocks from the reference
      size_t count = 0;
      bool is_converged = false;
      detail::set_norm(converge_test, this->norm(gradient));
      do {
        ++count;
        this->num_ALS++;

        for (size_t i = 0; i < ndim; i++) {
          this->direct(i, cur_block_size, fast_pI, matlab, converge_test, gradient);
          auto &ai = A[i];
          contract(this->one, ai, {1, 2}, ai.conj(), {1, 3}, this->zero, this->AtA[i], {2, 3});
        }

        is_converged = converge_test(A, this->AtA);
      }while(count < max_als && !is_converged);
      // Compute new difference
      gradient -= reconstruct(A, order, A[ndim]);

      // Copy the block computed in A to the correct portion in blockfactors
          // (this will replace A at the end)
      for(size_t i = 0; i < ndim; ++i){
        copy_blocks(blockfactors[i], A[i], block_start, block_end);
      }
      // Copy the lambda values into the correct location in blockfactors
      size_t c = 0;
      auto lam_full_ptr = blockfactors[ndim].data(), new_lam_ptr = A[ndim].data();
      for (auto b = block_start; b < block_end; ++b, ++c) {
        *(lam_full_ptr + b) = *(new_lam_ptr + c);
      }
    }

  };
}

#endif  // BTAS_GENERIC_CP_BCD_H
