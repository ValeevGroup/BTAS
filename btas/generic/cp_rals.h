//
// Created by Karl Pierce on 7/24/19.
//

#ifndef BTAS_GENERIC_CP_RALS_H
#define BTAS_GENERIC_CP_RALS_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <btas/generic/cp.h>
#include <btas/generic/rals_helper.h>

#ifdef BTAS_HAS_INTEL_MKL
#include <mkl_trans.h>
#endif

namespace btas {
  /** \brief Computes the Canonical Product (CP) decomposition of an order-N
    tensor using regularized alternating least squares (RALS).

    This computes the CP decomposition of btas::Tensor objects with row
    major storage only with fixed (compile-time) and variable (run-time)
    ranks. Also provides Tucker and randomized Tucker-like compressions coupled
    with CP-RALS decomposition. Does not support strided ranges.

   \warning this code takes a non-const reference \c tensor_ref but does
   not modify the values. This is a result of API (reshape needs non-const tensor)

    Synopsis:
    \code
    // Constructors
    CP_RALS A(tensor)                   // CP_ALS object with empty factor
                                        // matrices and no symmetries
    CP_RALS A(tensor, symms)            // CP_ALS object with empty factor
                                        // matrices and symmetries

    // Operations
    A.compute_rank(rank, converge_test)             // Computes the CP_RALS of tensor to
                                                    // rank, rank build and HOSVD options

    A.compute_rank_random(rank, converge_test)      // Computes the CP_RALS of tensor to
                                                    // rank. Factor matrices built at rank
                                                    // with random numbers

    A.compute_error(converge_test, omega)           // Computes the CP_RALS of tensor to
                                                    // 2-norm
                                                    // error < omega.

    A.compute_geometric(rank, converge_test, step)  // Computes CP_RALS of tensor to
                                                    // rank with
                                                    // geometric steps of step between
                                                    // guesses.

    A.compute_PALS(converge_test)           // computes CP_RALS of tensor to
                                                    // rank = 3 * max_dim(tensor)
                                                    // in 4 panels using a modified
                                                    // HOSVD initial guess

    A.compress_compute_tucker(tcut_SVD, converge_test) // Computes Tucker decomposition
                                                    // using
                                                    // truncated SVD method then
                                                    // computes finite
                                                    // error CP decomposition on core
                                                    // tensor.

    A.compress_compute_rand(rank, converge_test)    // Computes random decomposition on
                                                    // Tensor to
                                                    // make core tensor with every mode
                                                    // size rank
                                                    // Then computes CP decomposition
                                                    // of core.

   //See documentation for full range of options

    // Accessing Factor Matrices
    A.get_factor_matrices()             // Returns a vector of factor matrices, if
                                        // they have been computed

    A.reconstruct()                     // Returns the tensor computed using the
                                        // CP factor matrices
    \endcode
  */
  template <typename Tensor, class ConvClass = NormCheck<Tensor> >
  class CP_RALS : public CP_ALS<Tensor, ConvClass> {
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
    using CP_ALS<Tensor, ConvClass>::tensor_ref;
    using CP_ALS<Tensor, ConvClass>::size;

    /// Create a CP ALS object, child class of the CP object
    /// that stores the reference tensor.
    /// Reference tensor has no symmetries.
    /// \param[in] tensor the reference tensor to be decomposed.
    CP_RALS(Tensor &tensor) : CP_ALS<Tensor, ConvClass>(tensor) {
      for (size_t i = 0; i < ndim; ++i) {
        symmetries.push_back(i);
      }
    }

    /// Create a CP ALS object, child class of the CP object
    /// that stores the reference tensor.
    /// Reference tensor has symmetries.
    /// Symmetries should be set such that the higher modes index
    /// are set equal to lower mode indices (a 4th order tensor,
    /// where the second & third modes are equal would have a
    /// symmetries of {0,1,1,3}
    /// \param[in] tensor the reference tensor to be decomposed.
    /// \param[in] symms the symmetries of the reference tensor.
    CP_RALS(Tensor &tensor, std::vector<size_t> &symms)
        : CP<Tensor, ConvClass>(tensor.rank()) {
      symmetries = symms;
      if (symmetries.size() > ndim) BTAS_EXCEPTION("Too many symmetries provided")
      for (size_t i = 0; i < ndim; ++i) {
        if (symmetries[i] > i) BTAS_EXCEPTION("Symmetries should always refer to factors at earlier positions");
      }
    }

    ~CP_RALS() = default;

   protected:
    RALSHelper<Tensor> helper;  // Helper object to compute regularized steps

    /// performs the RALS method to minimize the loss function for a single rank
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
    /// return if \c fast_pI was successful.

    void ALS(ind_t rank, ConvClass &converge_test, bool dir, ind_t max_als, bool calculate_epsilon, double &epsilon,
             bool &fast_pI) {
      size_t count = 0;

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
          auto tmp = symmetries[i];
          if (tmp != i) {
            A[i] = A[tmp];
            lambda[i] = lambda[tmp];
          } else if (dir) {
            this->direct(i, rank, fast_pI, matlab, converge_test, lambda[i]);
          } else {
            update_w_KRP(i, rank, fast_pI, matlab, converge_test, lambda[i]);
          }
          // Compute the value s after normalizing the columns
          auto & ai = A[i];
          this->s = helper(i, ai);
          // recompute lambda
          lambda[i] = (lambda[i] * (this->s * this->s) / (s0 * s0)) * alpha + (1 - alpha) * lambda[i];
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

} // namespace btas

#endif //BTAS_GENERIC_CP_RALS_H
