//
// Created by Karl Pierce on 7/24/19.
//

#ifndef BTAS_GENERIC_CP_DF_ALS_H
#define BTAS_GENERIC_CP_DF_ALS_H

#include <btas/error.h>
#include <btas/generic/cp.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <btas/generic/default_random_seed.h>
#include <btas/generic/core_contract.h>
#include <btas/generic/flatten.h>
#include <btas/generic/khatri_rao_product.h>
#include <btas/generic/randomized.h>
#include <btas/generic/swap.h>
#include <btas/generic/tucker.h>
#include <btas/generic/converge_class.h>
#include <btas/generic/rals_helper.h>
#include <btas/generic/reconstruct.h>
#include <btas/generic/cp_als.h>

namespace btas {

  /** \brief Computes the Canonical Product (CP) decomposition of an order-N
    tensor where the tensor is represented as \f$ T = B^T Z \f$
    where \f$ B \in \mathbf{R}^{X \times I^1 \times I^2 \times \dots \times I^{n}} \f$
    and \f$ Z \in \mathbf{R}^{X \times I^{n+1} \times \dots \times I^N} \f$
    Here \f$ X \f$ is called the connected dimension, no factor matrix will be recovered
    for this mode.
    Decomposition optimization will use alternating least squares (ALS).

    \warning this code takes a non-const reference \c tensor_ref but does
   not modify the values. This is a result of API (reshape needs non-const tensor)

    Synopsis:
    \code
    // Constructors
    CP_DF_ALS A(B, Z)                   // CP_DF_ALS object with empty factor
                                        // matrices and no symmetries
    CP_DF_ALS A(B, Z, symms)            // CP_DF_ALS object with empty factor
                                        // matrices and symmetries

    // Operations
    A.compute_rank(rank, converge_test)             // Computes the CP_ALS of T tensor to
                                                    // rank, rank build and HOSVD options

    A.compute_rank_random(rank, converge_test)      // Computes the CP_ALS of T tensor to
                                                    // rank. Factor matrices built at rank
                                                    // with random numbers

    A.compute_error(converge_test, omega)           // Computes the CP_ALS of T tensor to
                                                    // 2-norm
                                                    // error < omega.

    A.compute_geometric(rank, converge_test, step)  // Computes CP_ALS of T tensor to
                                                    // rank with
                                                    // geometric steps of step between
                                                    // guesses.

    A.compute_PALS(converge_test)                   // computes CP_ALS of T tensor to
                                                    // rank = 3 * max_dim(tensor)
                                                    // in 4 panels using a modified
                                                    // HOSVD initial guess

   //See documentation for full range of options

    // Accessing Factor Matrices
    A.get_factor_matrices()             // Returns a vector of factor matrices, if
                                        // they have been computed

    A.reconstruct()                     // Returns the tensor T computed using the
                                        // CP factor matrices
    \endcode
  */
  template <typename Tensor, class ConvClass = NormCheck<Tensor>>
  class CP_DF_ALS : public CP<Tensor, ConvClass> {
   public:
    using CP<Tensor, ConvClass>::A;
    using CP<Tensor, ConvClass>::ndim;
    using CP<Tensor, ConvClass>::normCol;
    using CP<Tensor, ConvClass>::generate_KRP;
    using CP<Tensor, ConvClass>::generate_V;
    using CP<Tensor, ConvClass>::norm;
    using CP<Tensor, ConvClass>::symmetries;
    using typename CP<Tensor, ConvClass>::ind_t;
    using typename CP<Tensor, ConvClass>::ord_t;

    /// Create a CP DF ALS object, child class of the CP object
    /// that stores the reference tensors.
    /// Reference tensor has no symmetries.
    /// \param[in] left the reference tensor, \f$ B \f$ to be decomposed.
    /// \param[in] right the reference tensor, \f$ Z \f$  to be decomposed.
    CP_DF_ALS(Tensor &left, Tensor &right)
        : CP<Tensor, ConvClass>(left.rank() + right.rank() - 2)
        , tensor_ref_left(left)
        , tensor_ref_right(right)
        , ndimL(left.rank())
        , ndimR(right.rank()) {
      for (size_t i = 0; i < ndim; ++i) {
        symmetries.push_back(i);
      }
    }

    /// Create a CP ALS object, child class of the CP object
    /// that stores the reference tensors.
    /// Reference tensor has symmetries.
    /// Symmetries should be set such that the higher modes index
    /// are set equal to lower mode indices (a 4th order tensor,
    /// where the second & third modes are equal would have a
    /// symmetries of {0,1,1,3}
    /// \param[in] left the reference tensor, \f$ B \f$ to be decomposed.
    /// \param[in] right the reference tensor, \f$ Z \f$ to be decomposed.
    /// \param[in] symms the symmetries of the reference tensor.
    CP_DF_ALS(Tensor &left, Tensor &right, std::vector<size_t> &symms)
        : CP<Tensor, ConvClass>(left.rank() + right.rank() - 2)
        , tensor_ref_left(left)
        , tensor_ref_right(right)
        , ndimL(left.rank())
        , ndimR(right.rank()) {
      symmetries = symms;
      for (size_t i = 0; i < ndim; ++i) {
        if (symmetries[i] > i) BTAS_EXCEPTION("Symmetries should always refer to factors at earlier positions");
      }
      if (symmetries.size() != ndim)
        BTAS_EXCEPTION("Tensor describing symmetries must be equal to number of non-connected dimensions");
    }

    CP_DF_ALS() = default;

    ~CP_DF_ALS() = default;

    /// \brief Computes decomposition of the order-N tensor \c tensor
    /// with rank = \c RankStep * \c panels *  max_dim(reference_tensor) + max_dim(reference_tensor)
    /// Initial guess for factor matrices start at rank = max_dim(reference_tensor)
    /// and builds rank \c panel times by \c RankStep * max_dim(reference_tensor) increments

    /// \param[in, out] converge_list Tests to see if ALS is converged, holds the value of fit.
    /// should be as many tests as there are panels
    /// \param[in] RankStep CP_ALS increment of the panel
    /// \param[in] panels number of times the rank will be built
    /// \param[in]
    /// max_als Max number of iterations allowed to converge the ALS approximation default = 1e4
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = true
    /// \param[in]
    /// calculate_epsilon Should the 2-norm error be calculated \f$ ||T_{\rm exact} -
    /// T_{\rm approx}|| = \epsilon. \f$ Default = false.
    /// \param[in] direct Should the CP decomposition be computed without
    /// calculating the Khatri-Rao product? Default = true.
    /// \return  if ConvClass = FitCheck, returns the fit as defined by fitcheck
    /// else if calculate_epsilon = true, returns 2-norm error between exact and approximate tensor
    /// else return -1
    double compute_PALS(std::vector<ConvClass> &converge_list, double RankStep = 0.5, size_t panels = 4,
                        int max_als = 20, bool fast_pI = false, bool calculate_epsilon = false,
                        bool direct = true) override {
      if (RankStep <= 0) BTAS_EXCEPTION("Panel step size cannot be less than or equal to zero");
      if (converge_list.size() < panels)
        BTAS_EXCEPTION("Too few convergence tests.  Must provide a list of panels convergence tests");
      double epsilon = -1.0;
      size_t count = 0;
      // Find the largest rank this will be the first panel
      ind_t max_dim = tensor_ref_left.extent(0);
      for (size_t i = 1; i < ndimL; ++i) {
        ind_t dim = tensor_ref_left.extent(i);
        max_dim = (dim > max_dim ? dim : max_dim);
      }
      for (size_t i = 0; i < ndimR; ++i) {
        ind_t dim = tensor_ref_right.extent(i);
        max_dim = (dim > max_dim ? dim : max_dim);
      }

      while (count < panels) {
        auto converge_test = converge_list[count];
        // Use tucker initial guess (SVD) to compute the first panel
        if (count == 0) {
          build(max_dim, converge_test, direct, max_als, calculate_epsilon, 1, epsilon, true, max_dim, fast_pI);
          // build(max_dim, converge_test, max_als, calculate_epsilon, 1, epsilon, true, max_dim, fast_pI);
        }
        // All other panels build the rank buy RankStep variable
        else {
          // Always deal with the first matrix push new factors to the end of A
          // Kick out the first factor when it is replaced.
          // This is the easiest way to resize and preserve the columns
          // (if this is rebuilt with rank as columns this resize would be easier)
          ind_t rank = A[0].extent(1), rank_new = rank + RankStep * max_dim;
          for (size_t i = 0; i < ndim; ++i) {
            ind_t row_extent = A[0].extent(0), zero = 0;
            Tensor b(Range{Range1{A[0].extent(0)}, Range1{rank_new}});

            // Move the old factor to the new larger matrix
            {
              auto lower_old = {zero, zero}, upper_old = {row_extent, rank};
              auto old_view = make_view(b.range().slice(lower_old, upper_old), b.storage());
              auto A_itr = A[0].begin();
              for (auto iter = old_view.begin(); iter != old_view.end(); ++iter, ++A_itr) {
                *(iter) = *(A_itr);
              }
            }

            // Fill in the new columns of the factor with random numbers
            {
              auto lower_new = {zero, rank}, upper_new = {row_extent, rank_new};
              auto new_view = make_view(b.range().slice(lower_new, upper_new), b.storage());
              std::mt19937 generator(random_seed_accessor());
              std::uniform_real_distribution<> distribution(-1.0, 1.0);
              for (auto iter = new_view.begin(); iter != new_view.end(); ++iter) {
                *(iter) = distribution(generator);
              }
            }

            A.erase(A.begin());
            A.push_back(b);
            // replace the lambda matrix when done with all the factors
            if (i + 1 == ndim) {
              b.resize(Range{Range1{rank_new}});
              for (ind_t k = 0; k < A[0].extent(0); k++) b(k) = A[0](k);
              A.erase(A.begin());
              A.push_back(b);
            }
            // normalize the factor (don't replace the previous lambda matrix)
            normCol(0);
          }
          ALS(rank_new, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
        count++;
      }
      return epsilon;
    }

    /// \brief Computes decomposition of the order-N tensor \c tensor
    /// with rank = \c rank by first CP decomposing the component tensors
    /// \c tensor_ref_left and \c tensor_ref_right then uses the
    /// optimized factor matrices as initial guess to the compound decomposition.

    /// \param[in] rank Rank of the CP decomposition.
    /// \param[in, out] converge_test Tests to see if ALS is converged, holds the value of fit.
    /// \param[in]
    /// max_als Max number of iterations allowed to converge the ALS approximation default = 1e4
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = true
    /// \param[in]
    /// calculate_epsilon Should the 2-norm error be calculated \f$ ||T_{\rm exact} -
    /// T_{\rm approx}|| = \epsilon. \f$ Default = false.
    /// \param[in] direct Should the CP decomposition be computed without
    /// calculating the Khatri-Rao product? Default = true.
    /// \param[in] cp_comp_prec CP precision for the component subproblem decompositions
    /// Default = 1e-2.
    /// \return  if ConvClass = FitCheck, returns the fit as defined by fitcheck
    /// else if calculate_epsilon = true, returns 2-norm error between exact and approximate tensor
    /// else return -1
    double compute_comp_init(ind_t rank_cp3, ConvClass converge_test, size_t max_als = 1e4, bool fast_pI = true,
                            bool calculate_epsilon = false, bool direct = true, double cp_comp_prec = 1e-2, ind_t rank_cp4 = 0,
                             bool verbose = false) {
      rank_cp4 = (rank_cp4 == 0 ? rank_cp3 : rank_cp4);
      double epsilon = 0.0;
      auto nrm = [](Tensor &a) {
        auto norm = 0.0;
        for (auto &i : a) norm += i * i;
        return sqrt(norm);
      };
      if(rank_cp3 == rank_cp4) {
        // compute the left factor
        {
          FitCheck<Tensor> fit(cp_comp_prec);
          fit.set_norm(nrm(tensor_ref_left));
          fit.verbose(verbose);
          CP_ALS<Tensor, FitCheck<Tensor>> CP3(tensor_ref_left);
          auto error = CP3.compute_rank_random(rank_cp3, fit, 100, true);
          std::cout << "The accuracy of the LHS decomposition is : " << error * 100 << std::endl;
          init_factors_left = CP3.get_factor_matrices();
          auto cur_dim = init_factors_left.size() - 1;
          for (size_t i = 1; i < cur_dim; ++i) {
            A.emplace_back(init_factors_left[i]);
          }
        }
        // compute the right factor
        {
          FitCheck<Tensor> fit(cp_comp_prec);
          fit.set_norm(nrm(tensor_ref_right));
          fit.verbose(verbose);
          CP_ALS<Tensor, FitCheck<Tensor>> CP3(tensor_ref_right);
          auto error = CP3.compute_rank_random(rank_cp3, fit, 100, true);
          std::cout << "The accuracy of the RHS decomposition is : " << error * 100 << std::endl;
          init_factors_right = CP3.get_factor_matrices();
          auto cur_dim = init_factors_right.size();
          if (rank_cp3 == rank_cp4) {
            A.insert(A.end(), init_factors_right.begin() + 1, init_factors_right.end());
          }
        }
      } else{
        // fill the factors with random numbers
        std::mt19937 generator(random_seed_accessor());
        std::uniform_real_distribution<> distribution(-1.0, 1.0);
        for (size_t i = 1; i < ndimL; ++i) {
          auto &tensor_ref = tensor_ref_left;
          Tensor a(tensor_ref.extent(i), rank_cp4);
          for (auto iter = a.begin(); iter != a.end(); ++iter) {
            *(iter) = distribution(generator);
          }
          A.emplace_back(a);
        }
        for (size_t i = 1; i < ndimR; ++i) {
          auto &tensor_ref = tensor_ref_right;

          Tensor a(tensor_ref.extent(i), rank_cp4);
          for (auto iter = a.begin(); iter != a.end(); ++iter) {
            *(iter) = distribution(generator);
          }
          this->A.emplace_back(a);
        }
        Tensor lam(rank_cp4);
        lam.fill(1.0);
        A.emplace_back(lam);
        // compute the left factor and put in the CP4 tensor
        auto a_ptr = A.begin();
        {
          FitCheck<Tensor> fit(cp_comp_prec);
          fit.set_norm(nrm(tensor_ref_left));
          fit.verbose(verbose);
          CP_ALS<Tensor, FitCheck<Tensor>> CP3(tensor_ref_left);
          auto error = CP3.compute_rank_random(rank_cp3, fit, 100, true);
          std::cout << "LHS accuracy: " << error * 100 << std::endl;
          init_factors_left = CP3.get_factor_matrices();
          auto cur_dim = init_factors_left.size() - 1;
          for (size_t i = 1; i < cur_dim; ++i, ++a_ptr) {
            auto & tensor_ref = init_factors_left[i];
            auto left_ptr = tensor_ref.begin();
            auto row_dim = tensor_ref.extent(0);
            auto a_val_ptr = (*a_ptr).data();
            for(ind_t row = 0; row < row_dim; ++row){
              for(ind_t col = 0; col < rank_cp3; ++col, ++left_ptr){
                *(a_val_ptr + row * rank_cp4 + col) = *(left_ptr);
              }
            }
          }
        }
        // compute the right factor and put in the CP4 tensor
        {
          FitCheck<Tensor> fit(cp_comp_prec);
          fit.set_norm(nrm(tensor_ref_right));
          fit.verbose(verbose);
          CP_ALS<Tensor, FitCheck<Tensor>> CP3(tensor_ref_right);
          auto error = CP3.compute_rank_random(rank_cp3, fit, 100, true);
          std::cout << "RHS accuracy: " << error * 100 << std::endl;
          init_factors_right = CP3.get_factor_matrices();
          auto cur_dim = init_factors_right.size() - 1;
          for (size_t i = 1; i < cur_dim; ++i, ++a_ptr) {
            auto & tensor_ref = init_factors_right[i];
            auto right_ptr = tensor_ref.begin();
            auto row_dim = tensor_ref.extent(0);
            auto a_val_ptr = (*a_ptr).data();
            for(ind_t row = 0; row < row_dim; ++row){
              for(ind_t col = 0; col < rank_cp3; ++col, ++right_ptr){
                *(a_val_ptr + row * rank_cp4 + col) = *(right_ptr);
              }
            }
          }
        }
      }

      ALS(rank_cp4, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);

      detail::get_fit(converge_test, epsilon);
      return epsilon;
    }

    std::tuple<std::vector<Tensor>, std::vector<Tensor>> get_init_factors(){
      return std::make_tuple(init_factors_left, init_factors_right);
    }
   protected:
    Tensor &tensor_ref_left;   // Left connected tensor
    Tensor &tensor_ref_right;  // Right connected tensor
    size_t ndimL;              // Number of dimensions in left tensor
    size_t ndimR;              // number of dims in the right tensor
    bool lastLeft = false;
    Tensor leftTimesRight;
    std::vector<size_t> dims;
    std::vector<Tensor> init_factors_left;
    std::vector<Tensor> init_factors_right;

    /// Creates an initial guess by computing the SVD of each mode
    /// If the rank of the mode is smaller than the CP rank requested
    /// The rest of the factor matrix is filled with random numbers
    /// Builds factor matricies starting with R=(1 or SVD_rank)
    /// and moves to R = \c rank
    /// incrementing column dimension, R, by \c step

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit..
    /// \param[in] direct The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated \f$ ||T_{\rm exact} - T_{\rm approx}|| = \epsilon \f$ .
    /// \param[in] step
    /// CP_ALS built from r =1 to r = rank. r increments by step.
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in] SVD_initial_guess build inital guess from left singular vectors
    /// \param[in] SVD_rank rank of the initial guess using left singular vector
    /// \param[in,out] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// return if \c fast_pI was successful

    // TODO take advantage of symmetries in build
    void build(ind_t rank, ConvClass &converge_test, bool direct, ind_t max_als, bool calculate_epsilon, ind_t step,
               double &epsilon, bool SVD_initial_guess, ind_t SVD_rank, bool &fast_pI) override {
      {
        bool factors_set = false;
        // If its the first time into build and SVD_initial_guess
        // build and optimize the initial guess based on the left
        // singular vectors of the reference tensor.
        if (A.empty() && SVD_initial_guess) {
          if (SVD_rank == 0) BTAS_EXCEPTION("Must specify the rank of the initial approximation using SVD");
          make_svd_guess(SVD_rank);
          // Optimize this initial guess.
          ALS(SVD_rank, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
        // This loop keeps track of column dimension
        bool opt_in_for_loop = false;
        for (ind_t i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += step) {
          opt_in_for_loop = true;
          ind_t rank_new = i + 1;
          // This loop walks through the factor matrices
          for (size_t j = 0; j < ndim; ++j) {  // select a factor matrix
            // If no factor matrices exists, make a set of factor matrices
            // and fill them with random numbers that are column normalized
            // and create the weighting vector lambda
            if (i == 0) {
              Tensor a;
              if (j < ndimL - 1) {
                a = Tensor(Range{tensor_ref_left.range().range(j + 1), Range1{rank_new}});
              } else {
                a = Tensor(Range{tensor_ref_right.range().range(j - ndimL + 2), Range1{rank_new}});
              }
              std::mt19937 generator(random_seed_accessor());
              std::uniform_real_distribution<> distribution(-1.0, 1.0);
              for(auto iter = a.begin(); iter != a.end(); ++iter) {
                *(iter) = distribution(generator);
              }
              A.push_back(a);
              normCol(j);
            }

            // If the factor matrices have memory allocated, rebuild each matrix
            // with new column dimension col_dimension_old + skip
            // fill the new columns with random numbers and normalize the columns
            else {
              ind_t row_extent = A[0].extent(0), rank_old = A[0].extent(1), zero = 0;
              Tensor b(Range{A[0].range().range(0), Range1{rank_new}});

              {
                auto lower_old = {zero, zero}, upper_old = {row_extent, rank_old};
                auto old_view = make_view(b.range().slice(lower_old, upper_old), b.storage());
                auto A_itr = A[0].begin();
                for (auto iter = old_view.begin(); iter != old_view.end(); ++iter, ++A_itr) {
                  *(iter) = *(A_itr);
                }
              }

              {
                auto lower_new = {zero, rank_old}, upper_new = {row_extent, rank_new};
                auto new_view = make_view(b.range().slice(lower_new, upper_new), b.storage());
                std::mt19937 generator(random_seed_accessor());
                std::uniform_real_distribution<> distribution(-1.0, 1.0);
                for (auto iter = new_view.begin(); iter != new_view.end(); ++iter) {
                  *(iter) = distribution(generator);
                }
              }

              A.erase(A.begin());
              A.push_back(b);
              if (j + 1 == ndim) {
                A.erase(A.begin());
              }
            }
          }
          {
            Tensor lam(Range{Range1{rank_new}});
            A.push_back(lam);
          }
          // compute the ALS of factor matrices with rank = i + 1.
          ALS(rank_new, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
        if (factors_set && !opt_in_for_loop) {
          ALS(rank, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
      }
    }

    /// Generate the SVD initial guess of the super tensor
    /// without ever generating the super tensor using
    /// the HOSVD method. Here one flattens the tensor along
    /// the nth way then isolates that way by computing
    /// \f$ H_{n} = T^T_{n} T_{n} \f$ then computes the eigenvalue
    /// decomposition of H which provides the right singular vectors of \f$ T_n \f$

    /// \param[in] SVD_rank Initial guess rank, if SVD_rank is greater than the
    /// dimension of a mode, the factor matrix will be padded with random vectors.
    void make_svd_guess(ind_t SVD_rank){
      std::vector<size_t> left_modes, right_modes, result_modes, modes_w_dim_LT_svd;
      // Look through and find modes where I need to add extra columns

      {
        for (size_t i = 1; i < ndimL; i++) {
          if (tensor_ref_left.extent(i) < SVD_rank) {
            modes_w_dim_LT_svd.push_back(i - 1);
          }
        }
        for (size_t i = 1; i < ndimR; i++) {
          if (tensor_ref_right.extent(i) < SVD_rank) {
            modes_w_dim_LT_svd.push_back(i + ndimL - 2);
          }
        }
      }

      Tensor XXp, contracted;

      // Compute tensor_ref_right^T tensor_ref_right to keep the coupling dimension.
      // then contract with left hand side
      {
        for (size_t i = 0; i < ndimR; ++i) {
          left_modes.push_back(i);
          right_modes.push_back(i);
        }
        right_modes[0] = ndimR;
        result_modes.push_back(0);
        result_modes.push_back(ndimR);
        contract(1.0, tensor_ref_right, left_modes, tensor_ref_right, right_modes, 0.0, XXp, result_modes);
        left_modes.clear(); right_modes.clear();
        for (size_t i = 0; i < ndimL; i++) {
          left_modes.push_back(i);
          right_modes.push_back(i);
        }
        right_modes[0] = ndim + 3;
        result_modes[1] = ndim + 3;
        contract(1.0, XXp, result_modes, tensor_ref_left, left_modes, 0.0, contracted, right_modes);
        right_modes[0] = 0;
      }

      // Compute the SVD tile for factors of tensor_ref_left
      {
        auto ptrr = right_modes.begin() + 1, ptrf = result_modes.begin();
        for (size_t i = 1; i < ndimL; ++i, ++ptrr) {
          Tensor tucker;
          *(ptrr) = ndim + 3;
          *(ptrf) = i;
          contract(1.0, contracted, left_modes, tensor_ref_left, right_modes, 0.0, tucker, result_modes);
          *(ptrr) = i;

          auto R = tucker.extent(0);
          Tensor lambda(R);
          // Find the Singular vectors of the matrix using eigenvalue decomposition
          eigenvalue_decomp(tucker, lambda);

          // Fill a factor matrix with the singular vectors with the largest corresponding singular
          // values
          lambda = Tensor(R, SVD_rank);
          lambda.fill(0.0);
          auto lower_bound = {0, 0};
          auto upper_bound = {R, ((R > SVD_rank) ? SVD_rank : R)};
          auto view = make_view(tucker.range().slice(lower_bound, upper_bound), tucker.storage());
          auto l_iter = lambda.begin();
          for (auto iter = view.begin(); iter != view.end(); ++iter, ++l_iter) {
            *(l_iter) = *(iter);
          }

          A.push_back(lambda);
        }
      }

      // compute tensor_ref_left^T tensor_ref_left to keep coupling dimension.
      // then contract with RHS
      {
        right_modes[0] = ndim + 3;
        result_modes[0] = 0;
        contract(1.0, tensor_ref_left, left_modes, tensor_ref_left, right_modes, 0.0, XXp, result_modes);
        left_modes.clear(); right_modes.clear();
        for (size_t i = 0; i < ndimR; ++i) {
          left_modes.push_back(i);
          right_modes.push_back(i);
        }

        right_modes[0] = ndim + 3;
        result_modes[1] = ndim + 3;
        contract(1.0, XXp, result_modes, tensor_ref_right, left_modes, 0.0, contracted, right_modes);
        right_modes[0] = 0;

        auto ptrr = right_modes.begin() + 1, ptrf = result_modes.begin();
        for (size_t i = 1; i < ndimR; ++i, ++ptrr) {
          Tensor tucker;
          *(ptrr) = ndim + 3;
          *(ptrf) = i;
          contract(1.0, contracted, left_modes, tensor_ref_right, right_modes, 0.0, tucker, result_modes);
          *(ptrr) = i;

          auto R = tucker.extent(0);
          Tensor lambda(R);
          // Find the Singular vectors of the matrix using eigenvalue decomposition
          eigenvalue_decomp(tucker, lambda);

          // Fill a factor matrix with the singular vectors with the largest corresponding singular
          // values
          lambda = Tensor(R, SVD_rank);
          lambda.fill(0.0);
          auto lower_bound = {0, 0};
          auto upper_bound = {R, ((R > SVD_rank) ? SVD_rank : R)};
          auto view = make_view(tucker.range().slice(lower_bound, upper_bound), tucker.storage());
          auto l_iter = lambda.begin();
          for (auto iter = view.begin(); iter != view.end(); ++iter, ++l_iter) {
            *(l_iter) = *(iter);
          }

          A.push_back(lambda);
        }
      }

      std::mt19937 generator(random_seed_accessor());
      std::uniform_real_distribution<> distribution(-1.0, 1.0);
      // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
      for (auto &i : modes_w_dim_LT_svd) {
        ind_t R = A[i].extent(0), zero = 0;
        auto lower_bound = {zero, R};
        auto upper_bound = {R, SVD_rank};
        auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
        for (auto iter = view.begin(); iter != view.end(); ++iter) {
          *(iter) = distribution(generator);
        }
      }

      // Normalize the columns of the factor matrices and
      // set the values al lambda, the weigt of each order 1 tensor
      Tensor lambda(Range{Range1{SVD_rank}});
      A.push_back(lambda);
      for (size_t i = 0; i < ndim; ++i) {
        normCol(A[i]);
      }
    }

    /// Create a rank \c rank initial guess using
    /// random numbers from a uniform distribution

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit..
    /// \param[in] direct The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated \f$ ||T_{\rm exact} - T_{\rm approx}|| = \epsilon \f$ .
    /// \param[in] step
    /// CP_ALS built from r =1 to r = rank. r increments by step.
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in] SVD_initial_guess build inital guess from left singular vectors
    /// \param[in] SVD_rank rank of the initial guess using left singular vector
    /// \param[in, out] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// return if \c fast_pI was successful
    void build_random(ind_t rank, ConvClass &converge_test, bool direct, ind_t max_als, bool calculate_epsilon,
                      double &epsilon, bool &fast_pI) override {
      std::mt19937 generator(random_seed_accessor());
      std::uniform_real_distribution<> distribution(-1.0, 1.0);
      for (size_t i = 1; i < ndimL; ++i) {
        auto &tensor_ref = tensor_ref_left;
        Tensor a(Range{Range1{tensor_ref.extent(i)}, Range1{rank}});
        for (auto iter = a.begin(); iter != a.end(); ++iter) {
          *(iter) = distribution(generator);
        }
        A.push_back(a);
      }
      for (size_t i = 1; i < ndimR; ++i) {
        auto &tensor_ref = tensor_ref_right;

        Tensor a(tensor_ref.extent(i), rank);
        for (auto iter = a.begin(); iter != a.end(); ++iter) {
          *(iter) = distribution(generator);
        }
        this->A.push_back(a);
      }

      Tensor lambda(rank);
      lambda.fill(0.0);
      this->A.push_back(lambda);
      for (size_t i = 0; i < ndim; ++i) {
        normCol(i);
      }

      ALS(rank, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
    }

    /// performs the ALS method to minimize the loss function for a single rank
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
    /// \param[in,out] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// return if \c fast_pI was successful
    void ALS(ind_t rank, ConvClass &converge_test, int max_als, bool calculate_epsilon, double &epsilon,
             bool &fast_pI) {
      size_t count = 0;
      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      Tensor MtKRP(A[ndim - 1].extent(0), rank);
      leftTimesRight = Tensor(1);
      leftTimesRight.fill(0.0);
      // std::cout << "count\tfit\tchange" << std::endl;
      while (count < max_als && !is_converged) {
        count++;
        this->num_ALS++;
        for (size_t i = 0; i < ndim; i++) {
          auto tmp = symmetries[i];
          if (tmp == i) {
            direct(i, rank, fast_pI, matlab, converge_test);
          } else if (tmp < i) {
            A[i] = A[tmp];
          } else {
            BTAS_EXCEPTION("Incorrectly defined symmetry");
          }
        }
        is_converged = converge_test(A);
      }

      // Checks loss function if required
      if (calculate_epsilon) {
        if (typeid(converge_test) == typeid(btas::FitCheck<Tensor>)) {
          detail::get_fit(converge_test, epsilon);
          epsilon = 1 - epsilon;
        }
      }
    }

    /// Computes an optimized factor matrix holding all others constant.
    /// No Khatri-Rao product computed, immediate contraction
    /// Does this by first contracting a factor matrix with the refrence tensor
    /// Then computes hadamard/contraction products along all other modes except n.

    /// Want A(I2, R)
    /// T(I1, I2, I3, I4) = B(X, I1, I2) Z(X, I3, I4)
    /// B(X, I1, I2) (Z(X, I3, I4) * A(I4, R)) = B(X, I1, I2) Z'(X, I3, R)
    /// B(X, I1, I2) (Z'(X, I3, R) (*) A(I3, R)) = B(X, I1, I2) Z'(X, R) (contract along I3, Hadamard along R)
    /// B(X, I1, I2) * Z'(X, R) = B'(I1, I2, R)
    /// B'(I1, I2, R) (*) A(I1, R) = B'(I2, R) = A(I2, R) * V(R, R)

    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    /// \param[in,out] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// return if computing \c fast_pI was successful.
    /// \param[in, out] matlab If \c fast_pI = true then try to solve VA = B instead of taking pseudoinverse
    /// in the same manner that matlab would compute the inverse.
    /// return \c matlab was successful
    /// \param[in, out] converge_test Test to see if ALS is converged, holds the value of fit. test to see if the ALS is converged
    void direct(size_t n, ind_t rank, bool &fast_pI, bool &matlab, ConvClass &converge_test) {
      // Determine if n is in the left or the right tensor
      bool leftTensor = n < (ndimL - 1);
      Tensor an(A[n].range());

      if (lastLeft != leftTensor) {
        dims = std::vector<size_t>((leftTensor ? tensor_ref_left.rank() : tensor_ref_right.rank()));
        Tensor K(tensor_ref_right.extent(0), rank);
        {
          lastLeft = leftTensor;
          // want the tensor without n if n is in the left tensor take the right one and vice versa
          auto &tensor_ref = leftTensor ? tensor_ref_right : tensor_ref_left;

          // How many dimension in this side of the tensor
          size_t ndimCurr = tensor_ref.rank();
          ord_t sizeCurr = tensor_ref.size();
          // save range for resize at the end.
          auto R = tensor_ref.range();

          // Start by contracting with the last dimension of tensor without n
          // This is important for picking the correct factor matrix
          // not for picking from tensor_ref
          int contract_dim_inter = leftTensor ? ndim - 1 : ndimL - 2;

          // This is for size of the dimension being contracted
          // picked from tensor_ref
          ind_t contract_size = tensor_ref.extent(ndimCurr - 1);

          // Make the intermediate that will be contracted then hadamard contracted
          // Also resize the tensor for gemm contraction
          Tensor contract_tensor(sizeCurr / contract_size, rank);
          tensor_ref.resize(Range{Range1{sizeCurr / contract_size}, Range1{contract_size}});

          // Contract out the last dimension
          gemm(blas::Op::NoTrans, blas::Op::NoTrans, 1.0, tensor_ref, A[contract_dim_inter], 0.0, contract_tensor);
          // Resize tensor_ref back to original size
          tensor_ref.resize(R);

          // This is the size of the LH dimension of contract_tensor
          sizeCurr /= tensor_ref.extent(ndimCurr - 1);
          // for A now choose the next factor matrix
          --contract_dim_inter;

          // Now want to hadamard contract all the dimension that aren't the connecting dimension
          for (size_t i = 0; i < ndimCurr - 2; ++i, --contract_dim_inter) {
            // The contract_size now starts at the second last dimension
            contract_size = tensor_ref.extent(ndimCurr - 2 - i);
            // Store the LH most dimension size in idx1
            ord_t idx1 = sizeCurr / contract_size;
            contract_tensor.resize(Range{Range1{idx1}, Range1{contract_size}, Range1{rank}});
            // After hadamard product middle dimension is gone
            Tensor temp(idx1, rank);
            temp.fill(0.0);

            const auto &a = A[contract_dim_inter];
            ord_t j_times_rank = 0, j_times_cont_rank = 0;
            for (ind_t j = 0; j < idx1; ++j, j_times_rank += rank) {
              auto *temp_ptr = temp.data() + j_times_rank;
              ord_t k_times_rank = 0;
              for (ind_t k = 0; k < contract_size; ++k, k_times_rank += rank) {
                const auto *contract_ptr = contract_tensor.data() + j_times_cont_rank + k_times_rank;
                const auto *A_ptr = a.data() + k_times_rank;
                for (ord_t r = 0; r < rank; ++r) {
                  *(temp_ptr + r) += *(contract_ptr + r) * *(A_ptr + r);
                }
              }
              j_times_cont_rank += k_times_rank;
            }
            // After hadamard contract reset contract_tensor with new product
            contract_tensor = temp;
            // Remove the contracted dimension from the current size.
            sizeCurr = idx1;
          }

          // set the hadamard contracted tensor to the intermediate K
          K = contract_tensor;
        }
        {
          // contract K with the other side tensor
          // Tensor_ref now can be the side that contains n
          Tensor &tensor_ref = leftTensor ? tensor_ref_left : tensor_ref_right;
          // Modifying the dimension of tensor_ref so store the range here to resize
          // after contraction.
          Range R = tensor_ref.range();
          // make the new factor matrix for after process

          // LH side of tensor after contracting (doesn't include rank or connecting dimension)
          ord_t LH_size = tensor_ref.size() / tensor_ref.extent(0);
          // Temp holds the intermediate after contracting out the connecting dimension
          // It will be set up to enter hadamard product loop
          leftTimesRight = Tensor(LH_size, rank);
          // resize tensor_ref to remove connecting dimension
          tensor_ref.resize(Range{Range1{tensor_ref.extent(0)}, Range1{LH_size}});

          gemm(blas::Op::Trans, blas::Op::NoTrans, 1.0, tensor_ref, K, 0.0, leftTimesRight);
          // resize tensor_ref back to original dimensions
          tensor_ref.resize(R);

          // std::vector<int> dims(tensor_ref.rank());
          for (size_t i = 1; i < tensor_ref.rank(); ++i) {
            dims[i - 1] = tensor_ref.extent(i);
          }
          dims[dims.size() - 1] = rank;
        }
      }

      Tensor contract_tensor = leftTimesRight;
      ord_t LH_size = contract_tensor.size() / rank;
      // If hadamard loop has to skip a dimension it is stored here.
      ord_t pseudo_rank = rank;
      // number of dimensions in tensor_ref
      size_t ndimCurr = leftTensor ? tensor_ref_left.rank() : tensor_ref_right.rank();
      // the dimension that is being hadamard contracted out.
      size_t contract_dim = ndimCurr - 2, nInTensor = leftTensor ? n : n - ndimL + 1,
             a_dim = leftTensor ? contract_dim : ndim - 1, offset = 0;

      // go through hadamard contract on all dimensions excluding rank (will skip one dimension)
      for (size_t i = 0; i < ndimCurr - 2; ++i, --contract_dim, --a_dim) {
        auto contract_size = dims[contract_dim];
        LH_size /= contract_size;
        contract_tensor.resize(Range{Range1{LH_size}, Range1{contract_size}, Range1{pseudo_rank}});
        Tensor temp(Range{Range1{LH_size}, Range1{pseudo_rank}});
        const auto &a = A[a_dim];

        temp.fill(0.0);

        // If the middle dimension is the mode not being contracted, I will move
        // it to the right hand side temp((size of tensor_ref/product of
        // dimension contracted, rank * mode n dimension)
        if (nInTensor == contract_dim) {
          pseudo_rank *= contract_size;
          offset = contract_size;
        }

        // If the code hasn't hit the mode of interest yet, it will contract
        // over the middle dimension and sum over the rank.
        else if (contract_dim > nInTensor) {
          ord_t j_times_rank = 0, j_times_cont_rank = 0;
          for (ind_t j = 0; j < LH_size; ++j, j_times_rank += pseudo_rank) {
            auto *temp_ptr = temp.data() + j_times_rank;
            ord_t k_times_rank = 0;
            for (ind_t k = 0; k < contract_size; ++k, k_times_rank += pseudo_rank) {
              const auto *contract_ptr = contract_tensor.data() + j_times_cont_rank + k_times_rank;
              const auto *A_ptr = a.data() + k_times_rank;
              for (ind_t r = 0; r < pseudo_rank; ++r) {
                *(temp_ptr + r) += *(contract_ptr + r) * *(A_ptr + r);
              }
            }
            j_times_cont_rank += k_times_rank;
          }
          contract_tensor = temp;
        }
        // If the code has passed the mode of interest, it will contract over
        // the middle dimension and sum over rank * mode n dimension
        else {
          ord_t j_times_rank = 0, j_times_cont_rank = 0;
          for (ind_t j = 0; j < LH_size; ++j, j_times_rank += pseudo_rank) {
            auto *temp_ptr = temp.data() + j_times_rank;
            ord_t k_times_prank = 0, k_times_rank = 0;
            for (ind_t k = 0; k < contract_size; ++k, k_times_prank += pseudo_rank, k_times_rank += rank) {
              const auto *A_ptr = a.data() + k_times_rank;
              ord_t l_times_rank = 0;
              for (ind_t l = 0; l < offset; ++l, l_times_rank += rank) {
                const auto *contract_ptr = contract_tensor.data() + j_times_cont_rank + k_times_prank + l_times_rank;
                for (ind_t r = 0; r < rank; ++r) {
                  *(temp_ptr + l * rank + r) += *(contract_ptr + r) * *(A_ptr + r);
                  // temp(j, l*rank + r) += contract_tensor(j,k,l*rank+r) * A[a_dim](k,r);
                }
              }
            }
            j_times_cont_rank += k_times_rank;
          }

          contract_tensor = temp;
        }
      }

      // If the mode of interest is the 0th mode, then the while loop above
      // contracts over all other dimensions and resulting temp is of the
      // correct dimension If the mode of interest isn't 0th mode, must contract
      // out the 0th mode here, the above algorithm can't perform this
      // contraction because the mode of interest is coupled with the rank
      if (nInTensor != 0) {
        ind_t contract_size = contract_tensor.extent(0);
        Tensor temp(Range{Range1{offset}, Range1{rank}});
        contract_tensor.resize(Range{Range1{contract_size}, Range1{offset}, Range1{rank}});
        temp.fill(0.0);
        const auto &a = A[a_dim];
        ord_t i_times_rank = 0, i_times_off_rank = 0;
        for (ind_t i = 0; i < contract_size; i++, i_times_rank += rank) {
          const auto *A_ptr = a.data() + i_times_rank;
          ord_t j_times_rank = 0;
          for (ind_t j = 0; j < offset; j++, j_times_rank += rank) {
            const auto *contract_ptr = contract_tensor.data() + i_times_off_rank + j_times_rank;
            auto *temp_ptr = temp.data() + j_times_rank;
            for (ord_t r = 0; r < rank; r++) {
              *(temp_ptr + r) += *(A_ptr + r) * *(contract_ptr + r);
            }
          }
          i_times_off_rank += j_times_rank;
        }
        contract_tensor = temp;
      }

      detail::set_MtKRP(converge_test, contract_tensor);
      // multiply resulting matrix temp by pseudoinverse to calculate optimized
      // factor matrix
      // t1 = std::chrono::high_resolution_clock::now();

      this->pseudoinverse_helper(n, fast_pI, matlab, contract_tensor);
      // t2 = std::chrono::high_resolution_clock::now();
      // time = t2 - t1;
      // gemm_wPI += time.count();

      // Normalize the columns of the new factor matrix and update
      normCol(contract_tensor);
      A[n] = contract_tensor;
    }
  };

} // namepsace btas

#endif //BTAS_GENERIC_CP_DF_ALS_H
