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
    CP_BCD(Tensor &tensor, ind_t block_size = 1) : CP_ALS<Tensor, ConvClass>(tensor), blocksize(block_size){
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
    CP_BCD(Tensor &tensor, std::vector<size_t> &symms, ind_t block_size = 1)
        : CP_ALS<Tensor, ConvClass>(tensor, symms), blocksize(block_size) {
    }

    CP_BCD() = default;

    ~CP_BCD() = default;

   protected:
    Tensor gradient, block_ref;           // Stores gradient of BCD for fitting.
    ind_t blocksize;           // Size of block gradient
    std::vector<Tensor> blockfactors;

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
      std::vector<size_t> order;
      for(auto i = 0; i < ndim; ++i){
        order.emplace_back(i);
      }

      block_ref = tensor_ref;

      // Number of full blocks with rank blocksize
      // plus one more block with rank rank % blocksize
      int n_full_blocks = int( rank / blocksize),
          last_blocksize = rank % blocksize;

      long block_step = 0, z = 0;
      this->AtA = std::vector<Tensor>(ndim);
      // copying all of A to block factors. Then we are going to take slices of A
      // that way we can leverage all of `CP_ALS` code without modification
      blockfactors = A;
      // this stores the factors from rank 0 to #blocks * blocksize
      std::vector<Tensor> current_grads(ndim);
      // Compute all the BCD of the full blocks
      for (long b = 0; b < n_full_blocks; ++b, block_step += blocksize) {
        // Take the b-th block of the factor matrix also take the
        // find the partial grammian for the blocked factors
        for (size_t i = 0; i < ndim; ++i){
          auto & a_mat = A[i];
          auto lower = {z, block_step}, upper = {long(a_mat.extent(0)), block_step + blocksize};
          a_mat = make_view(blockfactors[i].range().slice(lower, upper), blockfactors[i].storage());
          contract(this->one, a_mat, {1, 2}, a_mat.conj(), {1, 3}, this->zero, this->AtA[i], {2, 3});
        }

        // Do the ALS loop
        //CP_ALS::ALS(blocksize, converge_test, dir, max_als, calculate_epsilon, epsilon, fast_pI);
        // First do it manually so we know its right
        // Compute ALS of the bth block against the gradient
        // computed by subtracting the previous blocks from the reference
        size_t count = 0;
        bool is_converged = false;
        bool matlab = true;
        auto one_over_tref = 1.0 / norm(tensor_ref);
        auto fit = 1.0, change = 0.0;
        do {
          ++count;
          this->num_ALS++;

          for (size_t i = 0; i < ndim; i++) {
            this->direct(i, blocksize, fast_pI, matlab, converge_test, gradient);
                // BCD(i, block_step, block_step + blocksize, fast_pI, matlab, converge_test);
                auto &ai = A[i];
            contract(this->one, ai, {1, 2}, ai.conj(), {1, 3}, this->zero, this->AtA[i], {2, 3});
            // Copy the block computed in A to the correct portion in blockfactors
            // (this will replace A at the end)
            copy_blocks(blockfactors[i], A[i], block_step, block_step + blocksize);
          }
          // Copy the lambda values into the correct location in blockfactors
          size_t c = 0;
          for (auto b = block_step; b < block_step + blocksize; ++b, ++c) {
            blockfactors[ndim](b) = A[ndim](c);
          }
          // Test the system to see if converged. Doing the hard way first
          // To compute the accuracy fully compute CP approximation using blocks 0 through b.
          for (size_t i = 0; i < ndim; ++i) {
            auto & a_mat = blockfactors[i];
            auto lower = {z, block_step}, upper = {long(a_mat.extent(0)), block_step + blocksize};
            current_grads[i] = make_view(a_mat.range().slice(lower, upper), a_mat.storage());
          }

          auto temp = reconstruct(current_grads, order, blockfactors[ndim]);
          auto newfit = 1.0 - norm(temp - tensor_ref) * one_over_tref;
          std::cout << newfit << std::endl;
          change = abs(fit - newfit);
          fit = newfit;
          std::cout << fit << "\t" << change << std::endl;
          is_converged = (change < 0.001);
          if(is_converged) {
            gradient = tensor_ref - temp;
          }
        }while(count < max_als && !is_converged);
        std::cout << gradient << std::endl;
      }
      // Fix tensor_ref
      auto ptr = tensor_ref.begin();
      for(auto g_ptr = gradient.begin(); g_ptr != gradient.end(); ++g_ptr, ++ptr)
        *(ptr) = *(g_ptr);
    }

    void copy_blocks(Tensor & to, Tensor & from, ind_t block_start, ind_t block_end){
        auto tref_dim = to.extent(0);
        auto to_rank = to.extent(1), from_rank = from.extent(1);

        auto to_ptr = to.data(), from_ptr = from.data();
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

    void BCD(size_t n, ind_t block_start, ind_t block_end, bool &fast_pI, bool &matlab, ConvClass &converge_test, double lambda = 0.0) {
      // Determine if n is the last mode, if it is first contract with first mode
      // and transpose the product
      bool last_dim = n == ndim - 1;
      // product of all dimensions
      ord_t LH_size = size;
      size_t contract_dim = last_dim ? 0 : ndim - 1;
      ind_t offset_dim = tensor_ref.extent(n);
      ind_t brank = block_end - block_start;
      ind_t pseudo_rank = brank;

      // Store the dimensions which are available to hadamard contract
      std::vector<ind_t> dimensions;
      for (size_t i = last_dim ? 1 : 0; i < (last_dim ? ndim : ndim - 1); i++) {
        dimensions.push_back(tensor_ref.extent(i));
      }

      // Modifying the dimension of tensor_ref so store the range here to resize
      Range R = tensor_ref.range();

      // Resize the tensor which will store the product of tensor_ref and the first factor matrix
      Tensor temp = Tensor(size / tensor_ref.extent(contract_dim), brank);
      gradient.resize(
          Range{Range1{last_dim ? tensor_ref.extent(contract_dim) : size / tensor_ref.extent(contract_dim)},
                Range1{last_dim ? size / tensor_ref.extent(contract_dim) : tensor_ref.extent(contract_dim)}});

      // contract tensor ref and the first factor matrix
      gemm((last_dim ? blas::Op::Trans : blas::Op::NoTrans), blas::Op::NoTrans, this->one , (last_dim? gradient.conj():gradient), blockfactors[contract_dim].conj(), this->zero,
           temp);

      // Resize tensor_ref
      gradient.resize(R);
      // Remove the dimension which was just contracted out
      LH_size /= tensor_ref.extent(contract_dim);

      // n tells which dimension not to contract, and contract_dim says which dimension I am trying to contract.
      // If n == contract_dim then that mode is skipped.
      // if n == ndim - 1, my contract_dim = 0. The gemm transposes to make rank = ndim - 1, so I
      // move the pointer that preserves the last dimension to n = ndim -2.
      // In all cases I want to walk through the orders in tensor_ref backward so contract_dim = ndim - 2
      n = last_dim ? ndim - 2 : n;
      contract_dim = ndim - 2;

      while (contract_dim > 0) {
        // Now temp is three index object where temp has size
        // (size of tensor_ref/product of dimension contracted, dimension to be
        // contracted, rank)
        ord_t idx2 = dimensions[contract_dim],
              idx1 = LH_size / idx2;
        temp.resize(
            Range{Range1{idx1}, Range1{idx2}, Range1{pseudo_rank}});
        Tensor contract_tensor;
        //Tensor contract_tensor(Range{Range1{idx1}, Range1{pseudo_rank}});
        //contract_tensor.fill(0.0);
        const auto &a = blockfactors[(last_dim ? contract_dim + 1 : contract_dim)];
        // If the middle dimension is the mode not being contracted, I will move
        // it to the right hand side temp((size of tensor_ref/product of
        // dimension contracted, rank * mode n dimension)
        if (n == contract_dim) {
          pseudo_rank *= offset_dim;
        }

        // If the code hasn't hit the mode of interest yet, it will contract
        // over the middle dimension and sum over the rank.

        else if (contract_dim > n) {
          middle_contract(this->one, temp, a.conj(), this->zero, contract_tensor);
          temp = contract_tensor;
        }

        // If the code has passed the mode of interest, it will contract over
        // the middle dimension and sum over rank * mode n dimension
        else {
          middle_contract_with_pseudorank(this->one, temp, a.conj(), this->zero, contract_tensor);
          temp = contract_tensor;
        }

        LH_size /= idx2;
        contract_dim--;
      }
      n = last_dim ? n+1 : n;

      // If the mode of interest is the 0th mode, then the while loop above
      // contracts over all other dimensions and resulting temp is of the
      // correct dimension If the mode of interest isn't 0th mode, must contract
      // out the 0th mode here, the above algorithm can't perform this
      // contraction because the mode of interest is coupled with the rank
      if (n != 0) {
        ind_t idx1 = dimensions[0];
        temp.resize(Range{Range1{idx1}, Range1{offset_dim}, Range1{brank}});
        Tensor contract_tensor(Range{Range1{offset_dim}, Range1{brank}});
        contract_tensor.fill(0.0);

        const auto &a = blockfactors[(last_dim ? 1 : 0)];
        front_contract(this->one, temp, a.conj(), this->zero, contract_tensor);

        temp = contract_tensor;
      }

      // Add lambda to factor matrices if RALS
      if(lambda !=0){
        auto LamA = blockfactors[n];
        scal(lambda, LamA);
        temp += LamA;
      }

      // multiply resulting matrix temp by pseudoinverse to calculate optimized
      // factor matrix
      pseudoinverse_block(n, brank, fast_pI, matlab, temp, lambda);

      // Normalize the columns of the new factor matrix and update
      this->normCol(temp, block_start);
      auto ptr = temp.data();
      blockfactors[n] = temp;
      {
        auto tref_dim = tensor_ref.extent(n);
        auto A_ptr = A[n].data();
        auto rank = A[n].extent(1);
        auto temp_pos = 0;
        for (ind_t i = 0, skip = 0; i < tref_dim; ++i, skip += rank){
          for (auto b = block_start; b < block_end; ++b, ++temp_pos){
            *(A_ptr + skip + b) = *(ptr + temp_pos);
          }
        }
      }
    }

    /// Calculates the column norms of a matrix and saves the norm values into
    /// lambda tensor (last matrix in the A)

    /// \param[in, out] Mat The matrix whose column will be normalized, return
    /// \c Mat with all columns normalized

    void normCol(Tensor &Mat, ord_t block_start) {
      if (Mat.rank() > 2) BTAS_EXCEPTION("normCol with rank > 2 not yet supported");
      ind_t rank = Mat.extent(1),
            Nsize = Mat.extent(0);
      ord_t size = Mat.size();

      auto Mat_ptr = Mat.data();
      std::vector<T> lambda;
      for(auto i = 0; i < rank; ++i) lambda.emplace_back(T(0.0));

      auto lam_ptr = lambda.data();
      for (ord_t i = 0; i < size; ++i) {
        *(lam_ptr + i % rank) += *(Mat_ptr + i) * btas::impl::conj(*(Mat_ptr + i));
      }

      auto A_ptr = A[ndim].data() + block_start;
      for (ind_t i = 0; i < rank; ++i) {
        auto val = sqrt(*(lam_ptr + i));
        *(A_ptr + i) = val;
        val = (abs(val) < 1e-12 ? 0.0 : 1.0 / val);
        btas::scal(Nsize, val, (Mat_ptr + i), rank);
      }
    }

  };
}

#endif  // BTAS_GENERIC_CP_BCD_H
