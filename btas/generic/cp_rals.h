//
// Created by Karl Pierce on 7/24/19.
//

#ifndef BTAS_GENERIC_CP_RALS_H
#define BTAS_GENERIC_CP_RALS_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <btas/btas.h>
#include <btas/error.h>
#include <btas/generic/default_random_seed.h>
#include "core_contract.h"
#include "flatten.h"
#include "khatri_rao_product.h"
#include "randomized.h"
#include "swap.h"
#include "tucker.h"
#include "converge_class.h"
#include "rals_helper.h"
#include "reconstruct.h"

#ifdef _HAS_INTEL_MKL
#include <mkl_trans.h>
#endif

namespace btas{
  /** \brief Computes the Canonical Product (CP) decomposition of an order-N
  tensor using a regularized alternating least squares (RALS).

  This computes the CP decomposition of btas::Tensor objects with row
  major storage only with fixed (compile-time) and variable (run-time)
  ranks. Also provides Tucker and randomized Tucker-like compressions coupled
  with CP-ALS decomposition. Does not support strided ranges.

  Synopsis:
  \code
  // Constructors
  CP_RALS A(tensor)                    // CP_RALS object with empty factor
  matrices

  // Operations
  A.compute_rank(rank, converge_test)             // Computes the CP_RALS of tensor to
                                                  // rank.

  A.compute_error(converge_test, omega)           // Computes the CP_RALS of tensor to
                                                  // 2-norm
                                                  // error < omega.

  A.compute_geometric(rank, converge_test, step)  // Computes CP_RALS of tensor to
                                                  // rank with
                                                  // geometric steps of step between
                                                  // guesses.

  A.paneled_tucker_build(converge_test)           // computes CP_ALS of tensor to
                                                  // rank = 2 * max_dim(tensor)
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
  template <typename Tensor, class ConvClass = NormCheck <Tensor> >
  class CP_RALS : public CP<Tensor, ConvClass>
        {
public:
    using CP<Tensor,ConvClass>::A;
    using CP<Tensor,ConvClass>::ndim;
    using CP<Tensor,ConvClass>::pseudoInverse;
    using CP<Tensor,ConvClass>::normCol;
    using CP<Tensor,ConvClass>::generate_KRP;
    using CP<Tensor,ConvClass>::generate_V;
    using CP<Tensor,ConvClass>::norm;

    /// Constructor of object CP_RALS
    /// \param[in] tensor The tensor object to be decomposed
    CP_RALS(Tensor &tensor) : CP<Tensor, ConvClass>(tensor.rank()),
                              tensor_ref(tensor), size(tensor.size()) {

    }

    ~CP_RALS() = default;

#ifdef _HAS_INTEL_MKL

    /// Computes decomposition of the order-N tensor \c tensor
    /// to \f$ rank = Max_Dim(\c tensor) + \c RankStep * Max_Dim(\c tensor) * \c panels \f$
    /// Initial guess for factor matrices is the modified HOSVD (tucker initial guess)
    /// number of ALS minimizations performed is \c panels. To minimize global
    /// CP problem choose \f$ 0 < \c RankStep \leq ~1.0 \f$

    /// \param[in] converge_test Test to see if ALS is converged
    /// \param[in] RankStep how much the rank should grow in each panel
    /// with respect to the largest dimension of \c tensor. Default = 0.25
    /// \param[in] panels number of ALS minimizations/steps
    /// \param[in] symm is \c tensor is symmetric in the last two dimension? default = false
    /// \param[in] max_als Max number of iterations allowed to
    /// converge the ALS approximation. default = 20
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = true
    /// \param[in] calculate_epsilon Should the
    /// 2-norm error be calculated \f$ ||T_{exact} - T_{approx}|| = \epsilon \f$.
    /// Default = false.
    /// \param[in] direct Should the CP
    /// decomposition be computed without calculating the Khatri-Rao product?
    /// Default = true.

    /// \returns 2-norm error
    /// between exact and approximate tensor, -1.0 if calculate_epsilon = false,
    /// \f$ \epsilon \f$
    double compute_PALS(std::vector<ConvClass> & converge_list, double RankStep = 0.5, int panels = 4, bool symm = false,
                                int max_als = 20,bool fast_pI = true, bool calculate_epsilon = false, bool direct = true) override{
      if (RankStep <= 0) BTAS_EXCEPTION("Panel step size cannot be less than or equal to zero");
      double epsilon = -1.0;
      int count = 0;
      // Find the largest rank this will be the first panel
      auto max_dim = tensor_ref.extent(0);
      for(int i = 1; i < ndim; ++i){
        auto dim = tensor_ref.extent(i);
        max_dim = ( dim > max_dim ? dim : max_dim);
      }

      while(count < panels){
        auto converge_test = converge_list[count];
        // Use tucker initial guess (SVD) to compute the first panel
        if(count == 0) {
          build(max_dim, converge_test, direct, max_als, calculate_epsilon, 1, epsilon, true, max_dim, fast_pI, symm);
        }
          // All other panels build the rank buy RankStep variable
        else {
          // Always deal with the first matrix push new factors to the end of A
          // Kick out the first factor when it is replaced.
          // This is the easiest way to resize and preserve the columns
          // (if this is rebuilt with rank as columns this resize would be easier)
          int rank = A[0].extent(1), rank_new = rank +  RankStep * max_dim;
          for (int i = 0; i < ndim; ++i) {
            int row_extent = A[0].extent(0);
            Tensor b(Range{Range1{A[0].extent(0)}, Range1{rank_new}});

            // Move the old factor to the new larger matrix
            {
              auto lower_old = {0, 0}, upper_old = {row_extent, rank};
              auto old_view = make_view(b.range().slice(lower_old, upper_old), b.storage());
              auto A_itr = A[0].begin();
              for(auto iter = old_view.begin(); iter != old_view.end(); ++iter, ++A_itr){
                *(iter) = *(A_itr);
              }
            }

            // Fill in the new columns of the factor with random numbers
            {
              auto lower_new = {0, rank}, upper_new = {row_extent, rank_new};
              auto new_view = make_view(b.range().slice(lower_new, upper_new), b.storage());
              std::mt19937 generator(random_seed_accessor());
              std::uniform_real_distribution<> distribution(-1.0, 1.0);
              for(auto iter = new_view.begin(); iter != new_view.end(); ++iter){
                *(iter) = distribution(generator);
              }
            }

            A.erase(A.begin());
            A.push_back(b);
            // replace the lambda matrix when done with all the factors
            if (i + 1 == ndim) {
              b.resize(Range{Range1{rank_new}});
              for (int k = 0; k < A[0].extent(0); k++) b(k) = A[0](k);
              A.erase(A.begin());
              A.push_back(b);
            }
            // normalize the factor (don't replace the previous lambda matrix)
            normCol(0);
          }
          ALS(rank_new, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
        }
        count++;
      }
      std::cout << "Number of ALS iterations was " << this->num_ALS << std::endl;
      return epsilon;
    }


    /// \brief Computes an approximate core tensor using
    /// Tucker decomposition, e.g.
    ///  \f$ T(I_1 \dots I_N) \approx T(R_1 \dots R_N) U^{(1)} (R_1, I_1) \dots U^{(N)} (R_N, I_N) \f$
    /// where \f$ \mathrm{rank} R_1 \leq \mathrm{rank } I_1 \f$ , etc.
    /// Reference: <a href="http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7516088">
    /// here</a>. Using this approximation the CP decomposition is
    /// computed to either finite error or finite rank. Default settings
    /// calculate to finite error. Factor matrices from get_factor_matrices() are
    /// scaled by the Tucker transformations.
    /// \note This requires Intel MKL.

    /// \param[in] tcutSVD Truncation threshold for SVD of each mode in Tucker
    /// decomposition.
    /// \param[in] converge_test Test to see if ALS is converged
    /// \param[in] opt_rank Should the CP decomposition of tucker
    /// core tensor find the optimal rank with error < tcutCP? Default = true.
    /// \param[in] tcutCP How small epsilon must be to consider the CP
    /// decomposition converged. Default = 1e-2.
    /// \param[in] rank If finding CP
    /// decomposition to finite rank, define CP rank. Default 0 will throw error
    /// for compute_rank.
    /// \param[in] direct The CP decomposition be computed
    /// without calculating the Khatri-Rao product? Default = true.
    /// \param[in]
    /// calculate_epsilon Should the 2-norm error be calculated \f$ ||T_{exact} -
    /// T_{approx}|| = \epsilon \f$ . Default = true.
    /// \param[in] step CP_RALS built
    /// from r =1 to r = \c rank. r increments by \c step; default = 1.
    /// \param[in]
    /// max_rank The highest rank approximation computed before giving up on
    /// CP-ALS. Default = 1e4.
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS. Default = 1e4.
    /// \param[in] tcutALS How small difference in
    /// factor matrices must be to consider ALS of a single rank converged.
    /// Default = 0.1.
    /// \param[in] SVD_initial_guess Should the initial factor matrices be
    /// approximated with left singular values? default = false
    /// \param[in] SVD_rank if \c
    /// SVD_initial_guess is true specify the rank of the initial guess such that
    /// SVD_rank <= rank. default = true
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = false
    /// \returns 2-norm error
    /// between exact and approximate tensor, -1.0 if calculate_epsilon = false,
    /// \f$ \epsilon \f$
    double compress_compute_tucker(double tcutSVD, ConvClass & converge_test, bool opt_rank = true, double tcutCP = 1e-2, int rank = 0,
                                   bool direct = true, bool calculate_epsilon = true, int step = 1, int max_rank = 1e4,
                                   double max_als = 1e4, bool SVD_initial_guess = false,
                                   int SVD_rank = 0, bool fast_pI = false) {
      // Tensor compression
      std::vector<Tensor> transforms;
      tucker_compression(tensor_ref, tcutSVD, transforms);
      size = tensor_ref.size();
      double epsilon = -1.0;

      // CP decomposition
      if (opt_rank)
        epsilon = this->compute_error(converge_test, tcutCP, step, max_rank, SVD_initial_guess, SVD_rank, false, max_als, fast_pI,
                                direct);
      else
        epsilon = this->compute_rank(rank, converge_test, step, SVD_initial_guess, SVD_rank, false, max_als, fast_pI, calculate_epsilon, direct);

      // scale factor matrices
      for (int i = 0; i < ndim; i++) {
        Tensor tt(transforms[i].extent(0), A[i].extent(1));
        gemm(CblasNoTrans, CblasNoTrans, 1.0, transforms[i], A[i], 0.0, tt);
        A[i] = tt;
      }

      return epsilon;
    }

    /// \brief Computes an approximate core tensor using
    /// random projection, i.e.
    /// \f$ T(I_1 \dots I_N) \approx T(R_1 \dots R_N) U^{(1)} (R_1, I_1) \dots U^{(N)} (R_N, I_N) \f$
    /// where \f$ \mathrm{rank } R_1 \leq \mathrm{rank } I_1 \f$ , etc.

    /// Reference: <a href="https://arxiv.org/pdf/1703.09074.pdf">arXiv:1703.09074</a>
    /// Using this approximation the CP decomposition is computed to
    /// either finite error or finite rank.
    /// Default settings calculate to finite error.
    /// Factor matrices are scaled by randomized transformation.
    /// \note This requires Intel MKL.

    /// \param[in] desired_compression_rank The new dimension of each mode after
    /// randomized compression.
    /// /// \param[in] converge_test Test to see if ALS is converged.
    /// \param[in] oversampl Oversampling added to the
    /// desired_compression_rank required to provide a more optimal decomposition.
    /// Default = suggested = 10.
    /// \param[in] powerit Number of power iterations,
    /// as specified in the literature, to scale the spectrum of each mode.
    /// Default = suggested = 2.
    /// \param[in] opt_rank Should the CP decomposition
    /// of tucker core tensor find the optimal rank with error < tcutCP? Default =
    /// true.
    /// \param[in] tcutCP How small epsilon must be to consider the CP
    /// decomposition converged. Default = 1e-2.
    /// \param[in] rank If finding CP
    /// decomposition to finite rank, define CP rank. Default 0 will throw error
    /// for compute_rank.
    /// \param[in] direct Should the CP decomposition be
    /// computed without calculating the Khatri-Rao product? Default = true.
    /// \param[in] calculate_epsilon Should the 2-norm error be calculated
    /// \f$ ||T_exact - T_approx|| = \epsilon \f$. Default = true.
    /// \param[in] step
    /// CP_RALS built from r =1 to r = rank. r increments by step; default = 1.
    /// \param[in] max_rank The highest rank approximation computed before giving
    /// up on CP-ALS. Default = 1e5.
    /// \param[in] max_als If CP decomposition is to
    /// finite error, max_als is the highest rank approximation computed before
    /// giving up on CP-ALS. Default = 1e5.
    /// \param[in] tcutALS How small
    /// difference in factor matrices must be to consider ALS of a single rank
    /// converged. Default = 0.1.
    /// \param[in] SVD_initial_guess Should the initial factor matrices be
    /// approximated with left singular values? default = false
    /// \param[in] SVD_rank if \c
    /// SVD_initial_guess is true specify the rank of the initial guess such that
    /// SVD_rank <= rank. default = true
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = false
    /// \returns
    /// 2-norm error between exact and approximate tensor, -1.0 if
    /// calculate_epsilon = false, \f$ \epsilon \f$
    double compress_compute_rand(int desired_compression_rank, ConvClass & converge_test, int oversampl = 10, int powerit = 2,
                                 bool opt_rank = true, double tcutCP = 1e-2, int rank = 0, bool direct = true,
                                 bool calculate_epsilon = false, int step = 1, int max_rank = 1e5, double max_als = 1e5,
                                 bool SVD_initial_guess = false, int SVD_rank = 0, bool fast_pI = false) {
      std::vector<Tensor> transforms;
      randomized_decomposition(tensor_ref, transforms, desired_compression_rank, oversampl, powerit);
      size = tensor_ref.size();
      double epsilon = -1.0;

      if (opt_rank)
        epsilon = this->compute_error(converge_test, tcutCP, step, max_rank, SVD_initial_guess, SVD_rank, false, max_als, fast_pI, direct);
      else
        epsilon = this->compute_rank(rank, converge_test, step, SVD_initial_guess, SVD_rank, false, max_als, fast_pI, calculate_epsilon, direct);

      // scale factor matrices
      for (int i = 0; i < ndim; i++) {
        Tensor tt(transforms[i].extent(0), A[i].extent(1));
        gemm(CblasNoTrans, CblasNoTrans, 1.0, transforms[i], A[i], 0.0, tt);
        A[i] = tt;
      }

      return epsilon;
    }

#endif  //_HAS_INTEL_MKL

  protected:
    int size;
    Tensor & tensor_ref;
    RALSHelper<Tensor> helper;

    /// Can create an initial guess by computing the SVD of each mode
    /// If the rank of the mode is smaller than the CP rank requested
    /// The rest of the factor matrix is filled with random numbers
    /// Also build factor matricies starting with R=(1,provided factor rank, SVD_rank)
    /// and moves to R = \c rank
    /// incrementing column dimension, R, by step

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged
    /// \param[in] direct The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS. Default = 1e5.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated \f$ ||T_{\rm exact} - T_{\rm approx}|| = \epsilon \f$ .
    /// \param[in] step
    /// CP_ALS built from r =1 to r = rank. r increments by step.
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in] SVD_initial_guess build inital guess from left singular vectors
    /// \param[in] SVD_rank rank of the initial guess using left singular vector
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?

    void build(int rank, ConvClass & converge_test, bool direct, int max_als, bool calculate_epsilon, int step, double &epsilon,
               bool SVD_initial_guess, int SVD_rank, bool & fast_pI, bool symm) override{
      // If its the first time into build and SVD_initial_guess
      // build and optimize the initial guess based on the left
      // singular vectors of the reference tensor.
#ifdef _HAS_INTEL_MKL
      if (A.empty() && SVD_initial_guess) {
        if (SVD_rank == 0) BTAS_EXCEPTION("Must specify the rank of the initial approximation using SVD");

        std::vector<int> modes_w_dim_LT_svd;
        A = std::vector<Tensor>(ndim);

        // Determine which factor matrices one can fill using SVD initial guess
        for(int i = 0; i < ndim; i++){
          if(tensor_ref.extent(i) < SVD_rank){
            modes_w_dim_LT_svd.push_back(i);
          }
        }

        // Fill all factor matrices with their singular vectors,
        // because we contract X X^T (where X is reference tensor) to make finding
        // singular vectors an eigenvalue problem some factor matrices will not be
        // full rank;

        for(int i = 0; i < ndim; i++){
          int R = tensor_ref.extent(i);
          Tensor S(R,R), lambda(R);

          // Contract refrence tensor to make it square matrix of mode i
          gemm(CblasNoTrans, CblasTrans, 1.0, flatten(tensor_ref, i), flatten(tensor_ref, i), 0.0, S);

          // Find the Singular vectors of the matrix using eigenvalue decomposition
          auto info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', R, S.data(), R, lambda.data());
          if (info) BTAS_EXCEPTION("Error in computing the SVD initial guess");

          // Fill a factor matrix with the singular vectors with the largest corresponding singular
          // values
          lambda = Tensor(R, SVD_rank);
          lambda.fill(0.0);
          auto lower_bound = {0,0};
          auto upper_bound = {R, ((R > SVD_rank) ? SVD_rank : R)};
          auto view = make_view(S.range().slice(lower_bound, upper_bound), S.storage());
          auto l_iter = lambda.begin();
          for(auto iter = view.begin(); iter != view.end(); ++iter, ++l_iter){
            *(l_iter) = *(iter);
          }

          A[i] = lambda;
        }

        //srand(3);
        std::mt19937 generator(random_seed_accessor());
        std::uniform_real_distribution<> distribution(-1.0, 1.0);
        // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
        for(auto& i: modes_w_dim_LT_svd){
          int R = tensor_ref.extent(i);
          auto lower_bound = {0, R};
          auto upper_bound = {R, SVD_rank};
          auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
          for(auto iter = view.begin(); iter != view.end(); ++iter){
            *(iter) = distribution(generator);
          }
        }

        // Normalize the columns of the factor matrices and
        // set the values al lambda, the weigt of each order 1 tensor
        Tensor lambda(Range{Range1{SVD_rank}});
        A.push_back(lambda);
        for(auto i = 0; i < ndim; ++i){
          normCol(A[i]);
        }

        helper = RALSHelper<Tensor>(A);

        // Optimize this initial guess.
        ALS(SVD_rank, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
      }
#else  // _HAS_INTEL_MKL
      if (SVD_initial_guess) BTAS_EXCEPTION("Computing the SVD requires LAPACK");
#endif // _HAS_INTEL_MKL
      // This loop keeps track of column dimension
      for (auto i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += step) {
        // This loop walks through the factor matrices
        for (auto j = 0; j < ndim; ++j) {  // select a factor matrix
          // If no factor matrices exists, make a set of factor matrices
          // and fill them with random numbers that are column normalized
          // and create the weighting vector lambda
          if (i == 0) {
            Tensor a(Range{tensor_ref.range(j), Range1{i + 1}});
            a.fill(rand());
            A.push_back(a);
            normCol(j);
            if (j  == ndim - 1) {
              Tensor lam(Range{Range1{i + 1}});
              A.push_back(lam);
            }
          }

            // If the factor matrices have memory allocated, rebuild each matrix
            // with new column dimension col_dimension_old + skip
            // fill the new columns with random numbers and normalize the columns
          else {
            int row_extent = A[0].extent(0), rank_old = A[0].extent(1);
            Tensor b(Range{A[0].range(0), Range1{i + 1}});

            {
              auto lower_old = {0, 0}, upper_old = {row_extent, rank_old};
              auto old_view = make_view(b.range().slice(lower_old, upper_old), b.storage());
              auto A_itr = A[0].begin();
              for(auto iter = old_view.begin(); iter != old_view.end(); ++iter, ++A_itr){
                *(iter) = *(A_itr);
              }
            }

            {
              auto lower_new = {0, rank_old}, upper_new = {row_extent, (int) i+1};
              auto new_view = make_view(b.range().slice(lower_new, upper_new), b.storage());
              std::mt19937 generator(random_seed_accessor());
              std::uniform_real_distribution<> distribution(-1.0, 1.0);
              for(auto iter = new_view.begin(); iter != new_view.end(); ++iter){
                *(iter) = distribution(generator);
              }
            }

            A.erase(A.begin());
            A.push_back(b);
            if (j + 1 == ndim) {
              b.resize(Range{Range1{i + 1}});
              for (int k = 0; k < A[0].extent(0); k++) b(k) = A[0](k);
              A.erase(A.begin());
              A.push_back(b);
            }
          }
        }
        // If column is already normalized will return 1
        for(auto i = 0; i < ndim; ++i){
          normCol(i);
        }

        helper = RALSHelper<Tensor>(A);
        // compute the ALS of factor matrices with rank = i + 1.
        ALS(i + 1, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
      }
    }

    void build_random(int rank, ConvClass & converge_test, bool direct, int max_als, bool calculate_epsilon, double &epsilon,
                      bool & fast_pI, bool symm) override{
      std::mt19937 generator(random_seed_accessor());
      std::uniform_real_distribution<> distribution(-1.0, 1.0);
      for(int i = 0; i < this->ndim; ++i){
        Tensor a(tensor_ref.extent(i), rank);
        //std::uniform_int_distribution<unsigned int> distribution(0, std::numeric_limits<unsigned int>::max() - 1);
        for(auto iter = a.begin(); iter != a.end(); ++iter){
          *(iter) = distribution(generator);
        }
        this->A.push_back(a);
        this->normCol(i);
      }
      Tensor lambda(rank);
      lambda.fill(0.0);
      this->A.push_back(lambda);

      ALS(rank, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
    }

    /// performs the ALS method to minimize the loss function for a single rank
    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged
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
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?

    void ALS(int rank, ConvClass & converge_test, bool dir, int max_als, bool calculate_epsilon, double &epsilon, bool & fast_pI, bool symm) {
      auto count = 0;

      double s = 0.0;
      const auto s0 = 1.0;
      std::vector<double> lambda(ndim, 1.0);
      const auto alpha = 0.8;
      if(symm){
        A[ndim - 1] = A[ndim -2];
      }

      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      while(count < max_als && !is_converged){
        count++;
        this->num_ALS++;
        for (auto i = 0; i < ((symm) ? ndim - 1: ndim); i++) {
          if (dir)
            direct(i, rank, symm, fast_pI, matlab, lambda[i], s, converge_test);
          else
            update_w_KRP(i, rank, symm, fast_pI, lambda[i], s, converge_test);

          lambda[i] = (lambda[i] * (s * s) / (s0 * s0)) * alpha + (1 - alpha) * lambda[i];
        }
        if(symm){
          A[ndim - 1] = A[ndim - 2];
        }
        is_converged = converge_test(A);
      }

      // Checks loss function if required
      if (calculate_epsilon) {
        if (typeid(converge_test) == typeid(btas::FitCheck<Tensor>)) {
          detail::get_fit(converge_test, epsilon);
          epsilon = 1 - epsilon;
        } else{
          epsilon = this->norm(this->reconstruct() - tensor_ref);
        }
        std::cout << "epsilon : " << epsilon << std::endl;
      }
    }

    /// Calculates an optimized CP factor matrix using Khatri-Rao product
    /// intermediate
    /// \param[in] n The mode being optimized, all other modes held
    /// constant
    /// \param[in] rank The current rank, column dimension of the factor
    /// matrices
    /// iteration factor matrix
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    void update_w_KRP(int n, int rank, bool symm, bool & fast_pI, double lambda, double & s, ConvClass & converge_test) {
      Tensor temp(A[n].extent(0), rank);
      Tensor an(A[n].range());

#ifdef _HAS_INTEL_MKL

      // Computes the Khatri-Rao product intermediate
      auto KhatriRao = generate_KRP(n, rank, true);

      // moves mode n of the reference tensor to the front to simplify contraction
      swap_to_first(tensor_ref, n);
      std::vector<size_t> tref_indices, KRP_dims, An_indices;

      // resize the Khatri-Rao product to the proper dimensions
      for (int i = 1; i < ndim; i++) {
        KRP_dims.push_back(tensor_ref.extent(i));
      }
      KRP_dims.push_back(rank);
      KhatriRao.resize(KRP_dims);
      KRP_dims.clear();

      // build contraction indices to contract over correct modes
      An_indices.push_back(0);
      An_indices.push_back(ndim);
      tref_indices.push_back(0);
      for (int i = 1; i < ndim; i++) {
        tref_indices.push_back(i);
        KRP_dims.push_back(i);
      }
      KRP_dims.push_back(ndim);

      contract(1.0, tensor_ref, tref_indices, KhatriRao, KRP_dims, 0.0, temp, An_indices);

      // move the nth mode of the reference tensor back where it belongs
      swap_to_first(tensor_ref, n, true);

#else  // BTAS_HAS_CBLAS

      // without MKL program cannot perform the swapping algorithm, must compute
      // flattened intermediate
      gemm(CblasNoTrans, CblasNoTrans, 1.0, flatten(tensor_ref, n), generate_KRP(n, rank, true), 0.0, temp);
#endif
      {
        auto LamA = A[n];
        scal(lambda, LamA);
        temp += LamA;
      }

      if(typeid(converge_test) == typeid(btas::FitCheck<Tensor>)) {
        converge_test.set_MtKRP(temp);
      }
      // contract the product from above with the psuedoinverse of the Hadamard
      // produce an optimize factor matrix
      gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, pseudoInverse(n, rank, fast_pI, lambda), 0.0, an);

      // compute the difference between this new factor matrix and the previous
      // iteration
      //for (auto l = 0; l < rank; ++l) A[ndim](l) = normCol(an, l);
      normCol(an);

      // Compute the value s after normalizing the columns
      s = helper(n, an);

      // Replace the old factor matrix with the new optimized result
      A[n] = an;
    }

    /// Computes an optimized factor matrix holding all others constant.
    /// No Khatri-Rao product computed, immediate contraction
    // Does this by first contracting a factor matrix with the refrence tensor
    // Then computes hadamard/contraction products along all other modes except n.

    // Want A(I2, R)
    // T(I1, I2, I3, I4)
    // T(I1, I2, I3, I4) * A(I4, R) = T'(I1, I2, I3, R)
    // T'(I1, I2, I3, R) (*) A(I3, R) = T'(I1, I2, R) (contract along I3, Hadamard along R)
    // T'(I1, I2, R) (*) A(I1, R) = T'(I2, R) = A(I2, R)

    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    /// \param[in] symm does the reference tensor have symmetry in the last two modes
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition

    void direct(int n, int rank, bool symm, bool & fast_pI, bool & matlab, double & lambda, double & s, ConvClass & converge_test) {
      //std::chrono::duration<double> time = t2 - t1;

      // Determine if n is the last mode, if it is first contract with first mode
      // and transpose the product
      bool last_dim = n == ndim - 1;
      // product of all dimensions
      int LH_size = size;
      int contract_dim = last_dim ? 0 : ndim - 1;
      int offset_dim = tensor_ref.extent(n);
      int pseudo_rank = rank;

      // Store the dimensions which are available to hadamard contract
      std::vector<int> dimensions;
      for(int i = last_dim ? 1: 0; i < (last_dim ? ndim: ndim - 1); i++){
        dimensions.push_back(tensor_ref.extent(i));
      }

      // Modifying the dimension of tensor_ref so store the range here to resize
      Range R = tensor_ref.range();
      Tensor an(A[n].range());

      // Resize the tensor which will store the product of tensor_ref and the first factor matrix
      Tensor temp = Tensor(size / tensor_ref.extent(contract_dim), rank);
      tensor_ref.resize(Range{
              Range1{last_dim ? tensor_ref.extent(contract_dim) : size / tensor_ref.extent(contract_dim)},
              Range1{last_dim ? size / tensor_ref.extent(contract_dim) : tensor_ref.extent(contract_dim)}});

      //auto t1 = std::chrono::high_resolution_clock::now();
      // contract tensor ref and the first factor matrix
      gemm((last_dim ? CblasTrans : CblasNoTrans), CblasNoTrans, 1.0, tensor_ref, A[contract_dim], 0.0, temp);
      //auto t2 = std::chrono::high_resolution_clock::now();
      //std::chrono::duration<double> time = t2 - t1;
      //gemm_first += time.count();

      // Resize tensor_ref
      tensor_ref.resize(R);
      // Remove the dimension which was just contracted out
      LH_size /= tensor_ref.extent(contract_dim);

      // n tells which dimension not to contract, and contract_dim says which dimension I am trying to contract.
      // If n == contract_dim then that mode is skipped.
      // if n == ndim - 1, my contract_dim = 0. The gemm transposes to make rank = ndim - 1, so I
      // move the pointer that preserves the last dimension to n = ndim -2.
      // In all cases I want to walk through the orders in tensor_ref backward so contract_dim = ndim - 2
      n = last_dim ? ndim - 2: n;
      contract_dim = ndim - 2;

      while (contract_dim > 0) {
        // Now temp is three index object where temp has size
        // (size of tensor_ref/product of dimension contracted, dimension to be
        // contracted, rank)
        temp.resize(Range{Range1{LH_size / dimensions[contract_dim]}, Range1{dimensions[contract_dim]},
                          Range1{pseudo_rank}});
        Tensor contract_tensor(Range{Range1{temp.extent(0)}, Range1{temp.extent(2)}});
        contract_tensor.fill(0.0);
        // If the middle dimension is the mode not being contracted, I will move
        // it to the right hand side temp((size of tensor_ref/product of
        // dimension contracted, rank * mode n dimension)
        if (n == contract_dim) {
          pseudo_rank *= offset_dim;
        }

          // If the code hasn't hit the mode of interest yet, it will contract
          // over the middle dimension and sum over the rank.
        else if (contract_dim > n) {
          //t1 = std::chrono::high_resolution_clock::now();
          auto idx1 = temp.extent(0);
          auto idx2 = temp.extent(1);
          for(int i = 0; i < idx1; i++){
            auto * contract_ptr = contract_tensor.data() + i * rank;
            for(int j = 0; j < idx2; j++){
              const auto * temp_ptr = temp.data() + i * idx2 * rank + j * rank;

              const auto * A_ptr = A[(last_dim ? contract_dim + 1: contract_dim)].data() + j * rank;
              for(int r = 0; r < rank; r++){
                *(contract_ptr + r) += *(temp_ptr + r) * *(A_ptr + r);
              }
            }
          }
          //t2 = std::chrono::high_resolution_clock::now();
          //time = t2 - t1;
          //gemm_second += time.count();
          temp = contract_tensor;
        }

          // If the code has passed the mode of interest, it will contract over
          // the middle dimension and sum over rank * mode n dimension
        else {
          //t1 = std::chrono::high_resolution_clock::now();
          int idx1 = temp.extent(0), idx2 = temp.extent(1), offset = offset_dim;
          for(int i = 0; i < idx1; i++){
            auto * contract_ptr = contract_tensor.data() + i * pseudo_rank;
            for(int j = 0; j < idx2; j++){
              const auto * temp_ptr = temp.data() + i * idx2 * pseudo_rank + j * pseudo_rank;

              const auto * A_ptr = A[(last_dim ? contract_dim + 1: contract_dim)].data() + j * rank;
              for(int k = 0; k < offset; k++){
                for(int r = 0; r < rank; r++){
                  *(contract_ptr + k * rank + r) += *(temp_ptr + k * rank + r) * *(A_ptr + r);
                }
              }
            }
          }
          //t2 = std::chrono::high_resolution_clock::now();
          //time = t2 - t1;
          //gemm_third += time.count();
          temp = contract_tensor;
        }

        LH_size /= tensor_ref.extent(contract_dim);
        contract_dim--;
      }

      // If the mode of interest is the 0th mode, then the while loop above
      // contracts over all other dimensions and resulting temp is of the
      // correct dimension If the mode of interest isn't 0th mode, must contract
      // out the 0th mode here, the above algorithm can't perform this
      // contraction because the mode of interest is coupled with the rank
      if (n != 0) {
        //t1 = std::chrono::high_resolution_clock::now();
        temp.resize(Range{Range1{dimensions[0]}, Range1{dimensions[n]}, Range1{rank}});
        Tensor contract_tensor(Range{Range1{temp.extent(1)}, Range1{rank}});
        contract_tensor.fill(0.0);

        int idx1 = temp.extent(0), idx2 = temp.extent(1);
        for(int i = 0; i < idx1; i++){
          const auto * A_ptr = A[(last_dim ? 1 : 0)].data() + i * rank;
          for(int j = 0; j < idx2; j++){
            const auto * temp_ptr = temp.data() + i * idx2 * rank + j * rank;
            auto * contract_ptr = contract_tensor.data() + j * rank;
            for(int r = 0; r < rank; r++){
              *(contract_ptr + r) += *(A_ptr + r) * *(temp_ptr + r);
            }
          }
        }
        //t2 = std::chrono::high_resolution_clock::now();
        //time = t2 - t1;
        //gemm_fourth += time.count();
        temp = contract_tensor;
      }

      n = last_dim ? ndim - 1: n;
      auto LamA = A[n];
      scal(lambda, LamA);
      temp += LamA;
      // multiply resulting matrix temp by pseudoinverse to calculate optimized
      // factor matrix
      //t1 = std::chrono::high_resolution_clock::now();

      if(typeid(converge_test) == typeid(btas::FitCheck<Tensor>)) {
        converge_test.set_MtKRP(temp);
      }

#ifdef _HAS_INTEL_MKL
      if(fast_pI && matlab) {
        // This method computes the inverse quickly for a square matrix
        // based on MATLAB's implementation of A / B operator.
        btas::Tensor<int, DEFAULT::range, varray<int> > piv(rank);
        piv.fill(0);

        auto a = generate_V(n, rank, lambda);
        int LDB = temp.extent(0);
        auto info = LAPACKE_dgesv(CblasColMajor, rank, LDB, a.data(), rank, piv.data(), temp.data(), rank);
        if (info == 0) {
          an = temp;
        }
        else{
          // If inverse fails resort to the pseudoinverse
          std::cout << "Matlab square inverse failed revert to fast inverse" << std::endl;
          matlab = false;
        }
      }
      if(!fast_pI || ! matlab){
        gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, pseudoInverse(n, rank, fast_pI, lambda), 0.0, an);
      }
#else
      matlab = false;
      if(!fast_pI || !matlab){
        gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, pseudoInverse(n, rank, fast_pI, lambda), 0.0, an);
      }
#endif
      //t2 = std::chrono::high_resolution_clock::now();
      //time = t2 - t1;
      //gemm_wPI += time.count();


      // Normalize the columns of the new factor matrix and update
      normCol(an);

      // Compute S after column normalization for RALS
      s = helper(n, an);

      A[n] = an;
    }

  };

} // namespace btas

#endif //BTAS_GENERIC_CP_RALS_H
