//
// Created by Karl Pierce on 2/10/22.
//

#ifndef MPQC_TUCK_COMP_CP_ALS_IPP
#define MPQC_TUCK_COMP_CP_ALS_IPP

#include <btas/generic/cp_als.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <btas/generic/rals_helper.h>

namespace btas{
  /** \brief Computes the Canonical Product (CP) decomposition of an order-N
    tensor which has been transformed via HOSVD (or some other defined transformation)
    using alternating least squares (ALS).

  This computes the CP decomposition of btas::Tensor objects with row
          major storage only with fixed (compile-time) and variable (run-time)
              ranks Does not support strided ranges.

   \warning this code takes a non-const reference \c tensor_ref and does not
                                         modify the values. This is a result of API (reshape needs non-const tensor)

                                             Synopsis:
    \code
      // Constructors
      TUCKER_CP_ALS A(tensor)        // TUCKER_CP_ALS object with empty factor
                               // matrices and empty transformation matrices

      // Operations
      A.compute_rank(rank, converge_test)             // Computes the CP of a tensor to
                                           // rank \c rank by either building the rank or using HOSVD.

      A.compute_rank_random(rank, converge_test)      // Computes the CP of tensor to
                                                  // rank \c rank. Factor matrices built at \c rank
                                                  // with random numbers

      A.compute_error(converge_test, omega)           // Computes the CP_ALS of tensor to
                                             // 2-norm
                                             // error < omega by building the rank (HOSVD option available)

      A.compute_geometric(rank, converge_test, step)  // Computes CP of tensor to
                                                      // rank with
                                                      // geometric steps of step between
                                                      // guesses.

      A.compute_PALS(converge_test)                   // computes CP_ALS of tensor to
                                     // rank = 3 * max_dim(tensor)
                                     // in 4 panels using a modified
                                     // HOSVD initial guess

      //See documentation for full range of options

      // Accessing Factor Matrices
      A.get_factor_matrices()             // Returns a vector of factor matrices, if
                               // they have been computed

      A.reconstruct()                     // Returns the tensor computed using the
                       // CP factor matrices
    \endcode
                                         */
  template <typename Tensor, class ConvClass = NormCheck<Tensor> >
  class TUCKER_CP_ALS : public CP_ALS<Tensor, ConvClass> {
   protected:
    using CP_ALS<Tensor, ConvClass>::tensor_ref;
    using typename CP_ALS<Tensor, ConvClass>::ind_t;
    using typename CP_ALS<Tensor, ConvClass>::ord_t;
    using CP_ALS<Tensor, ConvClass>::size;

    using CP<Tensor, ConvClass>::A;
    using CP<Tensor, ConvClass>::ndim;
    using CP<Tensor, ConvClass>::AtA;

   public:
    /// Create a Tucker compressed CP ALS object
    /// that stores but does not modify the reference tensor \c tensor.
    /// Unless some other transformation is defined, computes the
    /// Tucker decomposition truncating singular vectors with singular values
    /// less than |tensor| * \c epsilon_tucker
    /// \param[in] tensor the reference tensor to be decomposed.
    /// \param[in] epsilon_tucker truncation parameter for tucker decomposition
    TUCKER_CP_ALS(Tensor & tensor, double epsilon_tucker = 1e-3) :CP_ALS<Tensor, ConvClass>(tensor), tcut_tucker(epsilon_tucker){
    }

    /// Set the tensor transformation
    /// require that the factors be dimension (modified_size, orig_size)
    /// also assume that since you have the transormations, the reference is already
    /// transformed
    /// \param[in] facs : set of tucker factor matrices
    /// \param[in] transform_core : should the "original" non tucker approximated tensor be constructed?
   void set_tucker_factors(std::vector<Tensor> facs, bool transform_core = false){
     BTAS_ASSERT(facs.size() == this->ndim)
     size_t num = 0;
     tucker_factors.reserve(ndim);
     // because the reference is transformed we need the untransformed as tensor_ref
     for(auto i : facs){
       BTAS_ASSERT(i.rank() == 2)
       BTAS_ASSERT(i.extent(0) == tensor_ref.extent(num))
       tucker_factors.emplace_back(i);
       ++num;
     }

     core_tensor = tensor_ref;
     if(transform_core)
      transform_tucker(false, tensor_ref, tucker_factors);
   }

   protected:
    std::vector<Tensor> tucker_factors, transformed_A;
    Tensor core_tensor;
    double tcut_tucker;
    size_t core_size;

    /// computes the CP ALS of the tensor \c tensor using the core tensor \c core_tensor
    /// stops when converge_test is satisfies. Stores the exact CP factors in A
    /// stores the transformed CP factors in transformed_A
    /// only one solver so dir isn't used, just an artifact of base class.
    /// \param[in] rank current rank of the decomposotion
    /// \param[in] converge_test ALS satisfactory condition checker.
    /// \param[in] dir not used in this function
    /// \param[in] max_als maximum number of ALS iterations
    /// \param[in] calculate_epsilon should epsilon be returned, disregarded if ConvClass = FitCheck
    /// \param[in, out] epsilon in: a double value is disregarded.
    /// out: if ConvClass = FitCheck || \c calculate_epsilon the 2-norm tensor
    /// error of the CP approximation else not modified.
    /// \param[in] fast_pI Should ALS use a faster version of pseudoinverse?
    void ALS(ind_t rank, ConvClass &converge_test, bool dir, int max_als, bool calculate_epsilon, double &epsilon,
                  bool &fast_pI) override{
      if(tucker_factors.empty()) {
        make_tucker_factors(tensor_ref, tcut_tucker, tucker_factors, false);
        core_tensor = tensor_ref;
        transform_tucker(true, core_tensor, tucker_factors);
      }

      core_size = core_tensor.size();
      size_t count = 0;

      bool is_converged = false;
      bool matlab = fast_pI;
      if(AtA.empty()) {
        AtA = std::vector<Tensor>(ndim);
        transformed_A = std::vector<Tensor>(ndim);
      }
      auto ptr_A = A.begin(), ptr_T = tucker_factors.begin(),
           ptr_AtA = AtA.begin(), ptr_tran = transformed_A.begin();
      for (size_t i = 0; i < ndim; ++i,++ptr_A, ++ptr_T, ++ptr_AtA, ++ptr_tran) {
        auto &a_mat = A[i];
        *ptr_AtA = Tensor();
        contract(1.0, *ptr_A, {1, 2}, *ptr_A, {1, 3}, 0.0, *ptr_AtA, {2, 3});
        Tensor trans;
        *ptr_tran = Tensor();
        contract(1.0, *ptr_T, {1,2}, *ptr_A, {2,3}, 0.0, *ptr_tran, {1,3});
      }

      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      do{
        count++;
        this->num_ALS++;
        for (size_t i = 0; i < ndim; i++) {
          core_ALS_solver(i, rank, fast_pI, matlab, converge_test);
          auto &ai = A[i];
          contract(1.0, ai, {1, 2}, ai, {1, 3}, 0.0, AtA[i], {2, 3});
          contract(1.0, tucker_factors[i], {1,2}, ai, {2,3},0.0, transformed_A[i], {1,3});
        }
        is_converged = converge_test(A, AtA);
      }while (count < max_als && !is_converged);

      detail::get_fit(converge_test, epsilon);
      epsilon = 1 - epsilon;
      // Checks loss function if required
      if (calculate_epsilon && typeid(converge_test) != typeid(btas::FitCheck<Tensor>)) {
        epsilon = this->norm(this->reconstruct() - tensor_ref);
      }
    }

    /// This is  solver for ALS, it computes the optimal factor assuming all others fixed.
    /// Does not compute khatri-rao product, instead uses the same algorithm as base classes
    /// direct algorithm.
    /// \param[in] n Current mode being optimized
    /// \param[in] rank rank of the decomposition
    /// \param[in] fast_pI Should ALS use a faster version of pseudoinverse?
    /// \param[in, out] matlab in: if cholesky failes use fast pseudoinverse? out: did fast pseudoinverse fail?
    /// \param[in, out] converge_test in: important to set matricized tensor times khatri rao (MttKRP) if using FitCheck
    /// otherwise not used. out: \c converge_test with MttKRP set.
    /// \param[in] lambda regularization parameter.
    void core_ALS_solver(size_t n, ind_t rank, bool &fast_pI, bool &matlab, ConvClass &converge_test, double lambda = 0.0) {
      // Determine if n is the last mode, if it is first contract with first mode
      // and transpose the product
      bool last_dim = n == ndim - 1;
      // product of all dimensions
      ord_t LH_size = core_size;
      size_t contract_dim = last_dim ? 0 : ndim - 1;
      ind_t offset_dim = core_tensor.extent(n);
      ind_t pseudo_rank = rank;

      // Store the dimensions which are available to hadamard contract
      std::vector<ind_t> dimensions;
      for (size_t i = last_dim ? 1 : 0; i < (last_dim ? ndim : ndim - 1); i++) {
        dimensions.push_back(core_tensor.extent(i));
      }

      // Modifying the dimension of tensor_ref so store the range here to resize
      Range R = core_tensor.range();
      //Tensor an(A[n].range());

      // Resize the tensor which will store the product of tensor_ref and the first factor matrix
      Tensor An = Tensor(size / core_tensor.extent(contract_dim), rank);
      core_tensor.resize(
          Range{Range1{last_dim ? core_tensor.extent(contract_dim) : size / core_tensor.extent(contract_dim)},
                Range1{last_dim ? size / core_tensor.extent(contract_dim) : core_tensor.extent(contract_dim)}});

      // contract tensor ref and the first factor matrix
      gemm((last_dim ? blas::Op::Trans : blas::Op::NoTrans),
           blas::Op::NoTrans, 1.0, core_tensor, transformed_A[contract_dim], 0.0, An);

      // Resize tensor_ref
      core_tensor.resize(R);
      // Remove the dimension which was just contracted out
      LH_size /= core_tensor.extent(contract_dim);

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
        An.resize(
            Range{Range1{idx1}, Range1{idx2}, Range1{pseudo_rank}});
        Tensor contract_tensor;
        //Tensor contract_tensor(Range{Range1{idx1}, Range1{pseudo_rank}});
        //contract_tensor.fill(0.0);
        const auto &a = transformed_A[(last_dim ? contract_dim + 1 : contract_dim)];
        // If the middle dimension is the mode not being contracted, I will move
        // it to the right hand side temp((size of tensor_ref/product of
        // dimension contracted, rank * mode n dimension)
        if (n == contract_dim) {
          pseudo_rank *= offset_dim;
        }

        // If the code hasn't hit the mode of interest yet, it will contract
        // over the middle dimension and sum over the rank.

        else if (contract_dim > n) {
          middle_contract(1.0, An, a, 0.0, contract_tensor);
          An = contract_tensor;
        }

        // If the code has passed the mode of interest, it will contract over
        // the middle dimension and sum over rank * mode n dimension
        else {
          middle_contract_with_pseudorank(1.0, An, a, 0.0, contract_tensor);
          An = contract_tensor;
        }

        LH_size /= idx2;
        contract_dim--;
      }
      n = last_dim ? n+1 : n;

      // If the mode of interest is the 0th mode, then the while loop above
      // contracts over all other dimensions and resulting An is of the
      // correct dimension If the mode of interest isn't 0th mode, must contract
      // out the 0th mode here, the above algorithm can't perform this
      // contraction because the mode of interest is coupled with the rank
      if (n != 0) {
        ind_t idx1 = dimensions[0];
        An.resize(Range{Range1{idx1}, Range1{offset_dim}, Range1{rank}});
        Tensor contract_tensor(Range{Range1{offset_dim}, Range1{rank}});
        contract_tensor.fill(0.0);

        const auto &a = transformed_A[(last_dim ? 1 : 0)];
        front_contract(1.0, An, a, 0.0, contract_tensor);

        An = contract_tensor;
      }

      // Add lambda to factor matrices if RALS
      if(lambda !=0){
        auto LamA = A[n];
        scal(lambda, LamA);
        An += LamA;
      }
      // before providing the Matricized tensor times khatri rao product
      // need to reverse tucker transformation of that mode.
      {
        Tensor temp;
        contract(1.0, tucker_factors[n], {1,2}, An, {1,3}, 0.0, temp, {2,3});
        An = temp;
      }
      // multiply resulting matrix An by pseudoinverse to calculate optimized
      // factor matrix
      detail::set_MtKRP(converge_test, An);

      // Temp is then rewritten with unnormalized new A[n] matrix
      this->pseudoinverse_helper(n, fast_pI, matlab, An);

      // Normalize the columns of the new factor matrix and update
      this->normCol(An);
      A[n] = An;
    }

  };

  /** \brief Computes the Canonical Product (CP) decomposition of an order-N
    tensor which has been transformed via HOSVD (or some other defined transformation)
    using regularized alternating least squares (RALS).

  This computes the CP decomposition of btas::Tensor objects with row
          major storage only with fixed (compile-time) and variable (run-time)
              ranks Does not support strided ranges.

   \warning this code takes a non-const reference \c tensor_ref and does not
              modify the values. This is a result of API (reshape needs non-const tensor)

                  Synopsis:
    \code
      // Constructors
      TUCKER_CP_RALS A(tensor)        // TUCKER_CP_RALS object with empty factor
                               // matrices and empty transformation matrices

      // Operations
      A.compute_rank(rank, converge_test)             // Computes the CP of a tensor to
                                           // rank \c rank by either building the rank or using HOSVD.

      A.compute_rank_random(rank, converge_test)      // Computes the CP of tensor to
                                                  // rank \c rank. Factor matrices built at \c rank
                                                  // with random numbers

      A.compute_error(converge_test, omega)           // Computes the CP_RALS of tensor to
                                             // 2-norm
                                             // error < omega by building the rank (HOSVD option available)

      A.compute_geometric(rank, converge_test, step)  // Computes CP of tensor to
                                                      // rank with
                                                      // geometric steps of step between
                                                      // guesses.

      A.compute_PALS(converge_test)                   // computes CP_RALS of tensor to
                                     // rank = 3 * max_dim(tensor)
                                     // in 4 panels using a modified
                                     // HOSVD initial guess

      //See documentation for full range of options

      // Accessing Factor Matrices
      A.get_factor_matrices()             // Returns a vector of factor matrices, if
                               // they have been computed

      A.reconstruct()                     // Returns the tensor computed using the
                       // CP factor matrices
    \endcode
              */
  template <typename Tensor, class ConvClass = NormCheck<Tensor> >
  class TUCKER_CP_RALS : public TUCKER_CP_ALS<Tensor, ConvClass>{
   protected:
    using CP_ALS<Tensor, ConvClass>::tensor_ref;
    using typename CP_ALS<Tensor, ConvClass>::ind_t;
    using typename CP_ALS<Tensor, ConvClass>::ord_t;
    using CP_ALS<Tensor, ConvClass>::size;

    using CP<Tensor, ConvClass>::A;
    using CP<Tensor, ConvClass>::ndim;
    using CP<Tensor, ConvClass>::AtA;

    using TUCKER_CP_ALS<Tensor, ConvClass> ::core_tensor;
    using TUCKER_CP_ALS<Tensor, ConvClass> ::tucker_factors;
    using TUCKER_CP_ALS<Tensor, ConvClass> ::transformed_A;

   public:
    /// Create a Tucker compressed CP ALS object
    /// that stores but does not modify the reference tensor \c tensor.
    /// Unless some other transformation is defined, computes the
    /// Tucker decomposition truncating singular vectors with singular values
    /// less than |tensor| * \c epsilon_tucker
    /// \param[in] tensor the reference tensor to be decomposed.
    /// \param[in] epsilon_tucker truncation parameter for tucker decomposition
    TUCKER_CP_RALS(Tensor & tensor, double epsilon_tucker) :TUCKER_CP_ALS<Tensor, ConvClass>(tensor, epsilon_tucker){
    }

   protected:
    RALSHelper<Tensor> helper;  // Helper object to compute regularized steps

    /// computes the CP ALS of the tensor \c tensor using the core tensor \c core_tensor
    /// stops when converge_test is satisfies. Stores the exact CP factors in A
    /// stores the transformed CP factors in transformed_A
    /// only one solver so dir isn't used, just an artifact of base class.
    /// \param[in] rank current rank of the decomposotion
    /// \param[in] converge_test ALS satisfactory condition checker.
    /// \param[in] dir not used in this function
    /// \param[in] max_als maximum number of ALS iterations
    /// \param[in] calculate_epsilon should epsilon be returned, disregarded if ConvClass = FitCheck
    /// \param[in, out] epsilon in: a double value is disregarded.
    /// out: if ConvClass = FitCheck || \c calculate_epsilon the 2-norm tensor
    /// error of the CP approximation else not modified.
    /// \param[in] fast_pI Should ALS use a faster version of pseudoinverse?
    void ALS(ind_t rank, ConvClass &converge_test, bool dir, ind_t max_als, bool calculate_epsilon, double &epsilon,
             bool &fast_pI) {
      size_t count = 0;

      if(tucker_factors.empty()) {
        make_tucker_factors(tensor_ref, this->tcut_tucker, tucker_factors, false);
        core_tensor = tensor_ref;
        transform_tucker(true, core_tensor, tucker_factors);
      }

      if(AtA.empty()) {
        AtA = std::vector<Tensor>(ndim);
        transformed_A = std::vector<Tensor>(ndim);
      }
      auto ptr_A = A.begin(), ptr_T = tucker_factors.begin(),
           ptr_AtA = AtA.begin(), ptr_tran = transformed_A.begin();
      for (size_t i = 0; i < ndim; ++i,++ptr_A, ++ptr_T, ++ptr_AtA, ++ptr_tran) {
        auto &a_mat = A[i];
        *ptr_AtA = Tensor();
        contract(1.0, *ptr_A, {1, 2}, *ptr_A, {1, 3}, 0.0, *ptr_AtA, {2, 3});
        Tensor trans;
        *ptr_tran = Tensor();
        contract(1.0, *ptr_T, {1,2}, *ptr_A, {2,3}, 0.0, *ptr_tran, {1,3});
      }

      helper = RALSHelper<Tensor>(A);
      const auto s0 = 1.0;
      std::vector<double> lambda(ndim, 1.0);
      const auto alpha = 0.8;

      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      while (count < max_als && !is_converged) {
        count++;
        this->num_ALS++;
        for (size_t i = 0; i < ndim; i++) {
          this->direct(i, rank, fast_pI, matlab, converge_test, lambda[i]);
          // Compute the value s after normalizing the columns
          auto & ai = A[i];
          this->s = helper(i, ai);
          // recompute lambda
          lambda[i] = (lambda[i] * (this->s * this->s) / (s0 * s0)) * alpha + (1 - alpha) * lambda[i];
          contract(1.0, tucker_factors[i], {1,2}, ai, {2,3},0.0, transformed_A[i], {1,3});
          contract(1.0, ai, {1,2}, ai, {1,3}, 0.0, AtA[i], {2,3});
        }
        is_converged = converge_test(A);
      }

      // Checks loss function if required
      detail::get_fit(converge_test, epsilon);
      epsilon = 1 - epsilon;
      // Checks loss function if required
      if (calculate_epsilon && typeid(converge_test) != typeid(btas::FitCheck<Tensor>)) {
        epsilon = this->norm(this->reconstruct() - tensor_ref);
      }
    }
  };
}//namespace btas

#endif  // MPQC_TUCK_COMP_CP_ALS_IPP
