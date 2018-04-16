#ifndef BTAS_GENERIC_CP_RALS_H
#define BTAS_GENERIC_CP_RALS_H

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <btas/btas.h>
#include <btas/error.h>
#include "core_contract.h"
#include "flatten.h"
#include "khatri_rao_product.h"
#include "randomized.h"
#include "swap.h"
#include "tucker.h"

namespace btas {

  /** \brief Computes the Canonical Product (CP) decomposition of an order-N
    tensor using alternating least squares (ALS).

    This computes the CP decomposition of btas::Tensor objects with row
    major storage only with fixed (compile-time) and variable (run-time)
    ranks. Also provides Tucker and randomized Tucker-like compressions coupled
    with CP-ALS decomposition. Does not support strided ranges.

    Synopsis:
    \code
    // Constructors
    CP_ALS A(tensor)                    // CP_ALS object with empty factor
    matrices

    // Operations
    A.compute_rank(rank)                       // Computes the CP_ALS of tensor to
                                               // rank.

    A.compute_error(omega)                     // Computes the CP_ALS of tensor to
                                               // 2-norm
                                               // error < omega.

    A.compute_geometric(rank, step)            // Computes CP_ALS of tensor to
                                               // rank with
                                               // geometric steps of step between
                                               // guesses.

    A.compress_compute_tucker(tcut_SVD)        // Computes Tucker decomposition
                                               // using
                                               // truncated SVD method then
                                               // computes finite
                                               // error CP decomposition on core
                                               // tensor.

    A.compress_compute_rand(rank)              // Computes random decomposition on
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
  template <typename Tensor>
  class CP_RALS {
   public:
    /// Constructor of object CP_RALS
    /// \param[in] tensor The tensor object to be decomposed

    CP_RALS(Tensor &tensor) : tensor_ref(tensor), ndim(tensor_ref.rank()), size(tensor_ref.size()) {
#if not defined(BTAS_HAS_CBLAS) || not defined(_HAS_INTEL_MKL)
      BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__, "CP_ALS requires LAPACKE or mkl_lapack");
#endif
#ifdef _HAS_INTEL_MKL
#include <mkl_trans.h>
#endif
    }

    ~CP_RALS() = default;

    /// Computes decomposition of the order-N tensor \c tensor
    /// with CP rank = \c rank .
    /// Initial guess for factor matrices start at rank = 1
    /// and build to rank = \c rank by increments of \c step, to minimize
    /// error.

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] direct Should the CP decomposition be computed without
    /// calculating the Khatri-Rao product? Default = true.
    /// \param[in]
    /// calculate_epsilon Should the 2-norm error be calculated \f$ ||T_{\rm exact} -
    /// T_{\rm approx}|| = \epsilon. \f$ Default = false.
    /// \param[in] step CP_ALS built
    /// from r =1 to r = \c rank. r increments by \c step; default = 1.
    /// \param[in]
    /// max_als Max number of iterations allowed to converge the ALS approximation
    /// \param[in] tcutALS How small difference in factor matrices must be to
    /// consider ALS of a single rank converged. Default = 0.1.
    /// \param[in] SVD_initial_guess Should the initial factor matrices be
    /// approximated with left singular values?
    /// \param[in] SVD_rank if \c
    /// SVD_initial_guess is true specify the rank of the initial guess such that
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?
    /// \returns 2-norm
    /// error between exact and approximate tensor, -1 if calculate_epsilon =
    /// false.

    double compute_rank(int rank, bool direct = true, bool calculate_epsilon = false, int step = 1, int max_als = 1e5,
                        double tcutALS = 0.1, bool SVD_initial_guess = false, int SVD_rank = 0, bool symm = false) {
      if (rank <= 0) BTAS_EXCEPTION("Decomposition rank must be greater than 0");
      if (SVD_initial_guess && SVD_rank > rank) BTAS_EXCEPTION("Initial guess is larger than the desired CP rank");
      double epsilon = -1.0;
      build(rank, direct, max_als, calculate_epsilon, step, tcutALS, epsilon, SVD_initial_guess, SVD_rank, symm);
      return epsilon;
    }

    /// Computes the decomposition of the order-N tensor \c tensor
    /// to \f$ rank \leq \f$ \c max_als such that
    /// \f[ || T_{exact} - T_{approx}||_F = \epsilon \leq tcutCP \f]
    /// with rank incrementing by \c step.

    /// \param[in] tcutCP How small \f$\epsilon\f$ must be to consider the CP
    /// decomposition converged. Default = 1e-2.
    /// \param[in] direct Should the
    /// CP decomposition be computed without calculating the
    /// Khatri-Rao product? Default = true.
    /// \param[in] step CP_ALS built from r =1 to r = \c rank. r
    /// increments by \c step; default = 1.
    /// \param[in] max_rank The highest rank
    /// approximation computed before giving up on CP-ALS. Default = 1e5.
    /// \param[in] max_als Max number of iterations allowed to converge the ALS
    /// approximation
    /// \param[in] tcutALS How small difference in factor matrices
    /// must be to consider ALS of a single rank converged. Default = 0.1.
    /// \param[in] SVD_initial_guess Should the initial factor matrices be
    /// approximated with left singular values?
    /// \param[in] SVD_rank if \c
    /// SVD_initial_guess is true specify the rank of the initial guess such that
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?
    /// \returns 2-norm error
    /// between exact and approximate tensor, \f$ \epsilon \f$

    double compute_error(double tcutCP = 1e-2, bool direct = true, int step = 1, int max_rank = 1e5,
                         double max_als = 1e5, double tcutALS = 0.1, bool SVD_initial_guess = false, int SVD_rank = 0, bool symm = false) {
      int rank = (A.empty()) ? ((SVD_initial_guess) ? SVD_rank : 1) : A[0].extent(0);
      double epsilon = tcutCP + 1;
      while (epsilon > tcutCP && rank < max_rank) {
        build(rank, direct, max_als, true, step, tcutALS, epsilon, SVD_initial_guess, SVD_rank, symm);
        rank++;
      }
      return epsilon;
    }

    /// Computes decomposition of the order-N tensor \c tensor
    /// with \f$ CP rank \leq \f$ \c desired_rank \n
    /// Initial guess for factor matrices start at rank = 1
    /// and build to rank = \c rank by geometric steps of \c geometric_step, to
    /// minimize error.

    /// \param[in] desired_rank Rank of CP decomposition, r, will build by
    /// geometric step until \f$ r \leq \f$ \c desired_rank.
    /// \param[in] geometric_step CP_ALS built from r =1 to r = \c rank. r increments by r *=
    /// \c geometric_step; default = 2.
    /// \param[in] direct Should the CP
    /// decomposition be computed without calculating the Khatri-Rao product?
    /// Default = true.
    /// \param[in] max_als Max number of iterations allowed to
    /// converge the ALS approximation.
    /// \param[in] calculate_epsilon Should the
    /// 2-norm error be calculated \f$ ||T_{exact} - T_{approx}|| = \epsilon \f$.
    /// Default = false.
    /// \param[in] tcutALS How small difference in factor
    /// matrices must be to consider ALS of a single rank converged. Default =
    /// 0.1.
    /// \param[in] SVD_initial_guess Should the initial factor matrices be
    /// approximated with left singular values?
    /// \param[in] SVD_rank if \c SVD_initial_guess is true, specify the rank of the initial guess such that
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?
    /// \returns 2-norm error
    /// between exact and approximate tensor, -1.0 if calculate_epsilon = false,
    /// \f$ \epsilon \f$
    double compute_geometric(int desired_rank, int geometric_step = 2, bool direct = true, int max_als = 1e5,
                             bool calculate_epsilon = false, double tcutALS = 0.1, bool SVD_initial_guess = false,
                             int SVD_rank = 0, bool symm = false) {
      if (geometric_step <= 0) {
        BTAS_EXCEPTION("The step size must be larger than 0");
      }
      if (SVD_initial_guess && SVD_rank > desired_rank) {
        BTAS_EXCEPTION("Initial guess is larger than desired CP rank");
      }
      double epsilon = -1.0;
      int rank = (SVD_initial_guess) ? SVD_rank : 1;

      while (rank <= desired_rank && rank < max_als) {
        build(rank, direct, max_als, calculate_epsilon, geometric_step, tcutALS, epsilon, SVD_initial_guess, SVD_rank, symm);
        if (geometric_step <= 1)
          rank++;
        else
          rank *= geometric_step;
      }
      return epsilon;
    }

#ifdef _HAS_INTEL_MKL

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
    /// \param[in] step CP_ALS built
    /// from r =1 to r = \c rank. r increments by \c step; default = 1.
    /// \param[in]
    /// max_rank The highest rank approximation computed before giving up on
    /// CP-ALS. Default = 1e5.
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS. Default = 1e5.
    /// \param[in] tcutALS How small difference in
    /// factor matrices must be to consider ALS of a single rank converged.
    /// Default = 0.1.
    /// \param[in] SVD_initial_guess Should the initial factor
    /// matrices be approximated with left singular values?
    /// \param[in] SVD_rank if
    /// \c SVD_initial_guess is true specify the rank of the initial guess such
    /// \returns 2-norm error
    /// between exact and approximate tensor, -1.0 if calculate_epsilon = false,
    /// \f$ \epsilon \f$
    double compress_compute_tucker(double tcutSVD, bool opt_rank = true, double tcutCP = 1e-2, int rank = 0,
                                   bool direct = true, bool calculate_epsilon = true, int step = 1, int max_rank = 1e5,
                                   double max_als = 1e5, double tcutALS = 0.1, bool SVD_initial_guess = false,
                                   int SVD_rank = 0) {
      // Tensor compression
      std::vector<Tensor> transforms;
      tucker_compression(tensor_ref, tcutSVD, transforms);
      size = tensor_ref.size();
      double epsilon = -1.0;

      // CP decomposition
      if (opt_rank)
        epsilon = compute_error(tcutCP, direct, step, max_rank, max_als, tcutALS, SVD_initial_guess, SVD_rank);
      else
        epsilon = compute_rank(rank, direct, calculate_epsilon, step, max_als, tcutALS, SVD_initial_guess, SVD_rank);

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
    /// CP_ALS built from r =1 to r = rank. r increments by step; default = 1.
    /// \param[in] max_rank The highest rank approximation computed before giving
    /// up on CP-ALS. Default = 1e5.
    /// \param[in] max_als If CP decomposition is to
    /// finite error, max_als is the highest rank approximation computed before
    /// giving up on CP-ALS. Default = 1e5.
    /// \param[in] tcutALS How small
    /// difference in factor matrices must be to consider ALS of a single rank
    /// converged. Default = 0.1.
    /// \param[in] SVD_initial_guess Should the initial
    /// factor matrices be approximated with left singular values?
    /// \param[in]
    /// SVD_rank if \c SVD_initial_guess is true specify the rank of the initial
    /// \returns
    /// 2-norm error between exact and approximate tensor, -1.0 if
    /// calculate_epsilon = false, \f$ \epsilon \f$
    double compress_compute_rand(int desired_compression_rank, int oversampl = 10, int powerit = 2,
                                 bool opt_rank = true, double tcutCP = 1e-2, int rank = 0, bool direct = true,
                                 bool calculate_epsilon = false, int step = 1, int max_rank = 1e5, double max_als = 1e5,
                                 double tcutALS = .1, bool SVD_initial_guess = false, int SVD_rank = 0) {
      std::vector<Tensor> transforms;
      randomized_decomposition(tensor_ref, transforms, desired_compression_rank, oversampl, powerit);
      size = tensor_ref.size();
      double epsilon = -1.0;

      if (opt_rank)
        epsilon = compute_error(tcutCP, direct, step, max_rank, max_als, tcutALS, SVD_initial_guess, SVD_rank);
      else
        epsilon = compute_rank(rank, direct, calculate_epsilon, step, max_als, tcutALS, SVD_initial_guess, SVD_rank);

      // scale factor matrices
      for (int i = 0; i < ndim; i++) {
        Tensor tt(transforms[i].extent(0), A[i].extent(1));
        gemm(CblasNoTrans, CblasNoTrans, 1.0, transforms[i], A[i], 0.0, tt);
        A[i] = tt;
      }

      return epsilon;
    }

#endif  //_HAS_INTEL_MKL

    /// returns the rank \c rank optimized factor matrices
    /// \return Factor matrices stored in a vector. For example, a order-3
    /// tensor has factor matrices in positions [0]-[2]. In [3] there is scaling
    /// factor vector of size \c rank
    /// \throw  Exception if the CP decomposition is
    /// not yet computed.

    std::vector<Tensor> get_factor_matrices() {
      if (!A.empty())
        return A;
      else
        BTAS_EXCEPTION("Attempting to return a NULL object. Compute CP decomposition first.");
    }

    /// Function that uses the factor matrices from the CP
    /// decomposition and reconstructs the
    /// approximated tensor
    /// \returns The tensor approxmimated from the factor
    /// matrices of the CP decomposition.
    /// \throws Exception if the CP decomposition is
    /// not yet computed.
    Tensor reconstruct() {
      if(A.empty())
        BTAS_EXCEPTION("Factor matrices have not been computed. You must first calculate CP decomposition.");

      // Find the dimensions of the reconstructed tensor
      std::vector<size_t> dimensions;
      for (int i = 0; i < ndim; i++) {
        dimensions.push_back(A[i].extent(0));
      }

      // Scale the first factor matrix, this choice is arbitrary
      auto rank = A[0].extent(1);
      for (int i = 0; i < rank; i++) {
        scal(A[0].extent(0), A[ndim](i), std::begin(A[0]) + i, rank);
      }

      // Make the Khatri-Rao product of all the factor matrices execpt the last dimension
      Tensor KRP = A[0];
      Tensor hold = A[0];
      for (int i = 1; i < A.size() - 2; i++) {
        khatri_rao_product(KRP, A[i], hold);
        KRP = hold;
      }

      // contract the rank dimension of the Khatri-Rao product with the rank dimension of
      // the last factor matrix. hold is now the reconstructed tensor
      hold = Tensor(KRP.extent(0), A[ndim - 1].extent(0));
      gemm(CblasNoTrans, CblasTrans, 1.0, KRP, A[ndim - 1], 0.0, hold);

      // resize the reconstructed tensor to the correct dimensions
      hold.resize(dimensions);

      // remove the scaling applied to the first factor matrix
      for (int i = 0; i < rank; i++) {
        scal(A[0].extent(0), 1/A[ndim](i), std::begin(A[0]) + i, rank);
      }
      return hold;
    }

    void testing_function(int num_of_tests, int mode_to_test, int rank){
      for(int i = 0; i < ndim; i++){
        A.push_back(Tensor(tensor_ref.extent(i), rank));
        for(auto iter = A[i].begin(); iter != A[i].end(); ++iter){
          *(iter) = rand();
        }
      }
      A.push_back(Tensor(rank));

      for(int i = 0; i < num_of_tests; i++){
        double test = 0;
        direct(mode_to_test, rank, test);
      }

    }

   private:
    std::vector<Tensor> A;  // The vector of factor matrices
    Tensor &tensor_ref;     // The reference tensor being decomposed
    const int ndim;         // Number of modes in the reference tensor
    int size;               // Number of elements in the reference tensor

    /// creates factor matricies starting with R=1 and moves to R = \c rank
    /// incrementing column dimension, R, by step

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] direct The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS. Default = 1e5.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated \f$ ||T_{\rm exact} - T_{\rm approx}|| = \epsilon \f$ .
    /// \param[in] step
    /// CP_ALS built from r =1 to r = rank. r increments by step.
    /// \param[in]
    /// tcutALS How small difference in factor matrices must be to consider ALS of
    /// a single rank converged. Default = 0.1.
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in] SVD_initial_guess build inital guess from left singular vectors
    /// \param[in] SVD_rank rank of the initial guess using left singular vector
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?

    void build(int rank, bool direct, int max_als, bool calculate_epsilon, int step, double tcutALS, double &epsilon,
               bool SVD_initial_guess, int SVD_rank, bool symm) {
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
          auto info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'L', R, S.data(), R, lambda.data());
          if (info) BTAS_EXCEPTION("Error in computing the SVD initial guess");

          // Fill a factor matrix with the singular vectors with the largest corresponding singular
          // values
          lambda = Tensor(R, SVD_rank);
          auto lower_bound = {0, ((R > SVD_rank) ? R - SVD_rank : 0)};
          auto upper_bound = {R, R};
          auto view = make_view(S.range().slice(lower_bound, upper_bound), S.storage());
          auto l_iter = lambda.begin();
          for(auto iter = view.begin(); iter != view.end(); ++iter, ++l_iter){
            *(l_iter) = *(iter);
          }

          A[i] = lambda;
        }

        // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
        for(auto& i: modes_w_dim_LT_svd){
          int R = tensor_ref.extent(i);
          auto lower_bound = {0, R};
          auto upper_bound = {R, SVD_rank};
          auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
          for(auto iter = view.begin(); iter != view.end(); ++iter){
            *(iter) = rand() % 500;
          }
        }

        // Normalize the columns of the factor matrices and
        // set the values al lambda, the weigt of each order 1 tensor
        Tensor lambda(Range{Range1{SVD_rank}});
        auto lambda_ptr = lambda.data();
        for(auto &i: A){
          for(int j = 0; j < SVD_rank; j++){
            *(lambda_ptr + j) = normCol(i,j);
          }
        }
        A.push_back(lambda);

        // Optimize this initial guess.
        ALS(SVD_rank, direct, max_als, calculate_epsilon, tcutALS, epsilon, symm);
      }
#else  //
      if (SVD_initial_guess) BTAS_EXCEPTION("Computing the SVD requires LAPACK");
#endif
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
            normCol(a, i);
            A.push_back(a);
            if (j + 1 == ndim) {
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
              // auto rand = rand();
              for(auto iter = new_view.begin(); iter != new_view.end(); ++iter){
                *(iter) = rand() % 500;
                //*(iter) = rand;
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
        // compute the ALS of factor matrices with rank = i + 1.
        ALS(i + 1, direct, max_als, calculate_epsilon, tcutALS, epsilon, symm);
      }
    }

    /// performs the ALS method to minimize the loss function for a single rank
    /// \param[in] rank The rank of the CP decomposition.
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
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?

    void ALS(int rank, bool dir, int max_als, bool calculate_epsilon, double tcutALS, double &epsilon, bool symm) {
      auto count = 0;
      double test = tcutALS + 1.0;

      double s = 0.0;
      const auto s0 = 70.0;
      const auto alpha = 0.2;
      std::vector<double> lambda(ndim, 1.0);
      if(symm){
        A[ndim - 1] = A[ndim -2];
      }
      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      while (count <= max_als && test > tcutALS) {
        count++;
        test = 0.0;
        s = 0.0;

        for (auto i = 0; i < ((symm) ? ndim - 1: ndim); i++) {
          if (dir)
            direct(i, rank, test, symm, lambda[i]);
          else
            update_w_KRP(i, rank, test, symm, lambda[i]);

          test += s;
          s /= norm(A[i]);
          lambda[i] = (lambda[i] * (s * s) / (s0 * s0) ) * alpha + (1 - alpha) * lambda[i];
        }
        if(symm){
          A[ndim - 1] = A[ndim - 2];
        }
      }

      // Checks loss function if required
      if (calculate_epsilon) {
        epsilon = norm(reconstruct() - tensor_ref);
      }
      //num_ALS += count;
    }

    /// Calculates an optimized CP factor matrix using Khatri-Rao product
    /// intermediate
    /// \param[in] n The mode being optimized, all other modes held
    /// constant
    /// \param[in] rank The current rank, column dimension of the factor
    /// matrices
    /// \param[in out] test The difference between previous and current
    /// iteration factor matrix
    /// \param[in] symm is \c tensor is symmetric in the last two dimension?
    void update_w_KRP(int n, int rank, double &test, bool symm, double lambda) {
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
      // contract the product from above with the psuedoinverse of the Hadamard
      // produce an optimize factor matrix
      gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, pseudoInverse(n, rank, lambda), 0.0, an);

      // compute the difference between this new factor matrix and the previous
      // iteration
      for (auto l = 0; l < rank; ++l) A[ndim](l) = normCol(an, l);
      auto nrm = norm(A[n] - an);
      if(n == ndim - 2 && symm){
        test += nrm;
      }
      test += nrm;

      // Replace the old factor matrix with the new optimized result
      A[n] = an;
    }

    // For debug purposes
    void print(Tensor & A){
      if(A.rank() == 2){
        for(int i = 0; i < A.extent(0); i++){
          for(int j = 0; j < A.extent(1); j++) {
            std::cout << A(i,j) << " ";
          }
          std::cout << std::endl;
        }

      }
      else
        for(auto& i: A)
          std::cout << i << std::endl;
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
    /// \param[in out] test The difference between previous and current iteration
    /// factor matrix
    void direct(int n, int rank, double &test, bool symm,  double lambda) {
      //auto t1 = std::chrono::high_resolution_clock::now();
      //auto t2 = std::chrono::high_resolution_clock::now();
      //std::chrono::duration<double> time = t2 - t1;

      // Determine if n is the last mode, if it is first contract with first mode
      // and transpose the product
      bool last_dim = n == ndim - 1;
      // product of all dimensions
      int LH_size = size;
      int contract_dim = last_dim ? 0 : ndim - 1;
      int offset_dim = tensor_ref.extent(n);
      int pseudo_rank = rank;
      //double first_gemm = 0, second_gemm = 0, third_gemm = 0, final_gemm = 0;  // These are for timings

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

      //t1 = std::chrono::high_resolution_clock::now();
      // contract tensor ref and the first factor matrix
      gemm((last_dim ? CblasTrans : CblasNoTrans), CblasNoTrans, 1.0, tensor_ref, A[contract_dim], 0.0, temp);
      //t2 = std::chrono::high_resolution_clock::now();
      //time = t2 - t1;
      //first_gemm = time.count();

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
          //second_gemm = time.count();
          temp = contract_tensor;
        }

          // If the code has passed the mode of interest, it will contract over
          // the middle dimension and sum over rank * mode n dimension
        else {
          //t1 = std::chrono::high_resolution_clock::now();
          int idx1 = temp.extent(0), idx2 = temp.extent(1), offset = offset_dim;
          for(int i = 0; i < idx1; i++){
            auto * contract_ptr = contract_tensor.data() + i * rank;
            for(int j = 0; j < idx2; j++){
              const auto * temp_ptr = temp.data() + i * idx2 * offset * rank + j * offset * rank;

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
        //third_gemm = time.count();
        temp = contract_tensor;
      }

      n = last_dim ? ndim - 1: n;
      auto LamA = A[n];
      scal(lambda, LamA);
      temp += LamA;
      // multiply resulting matrix temp by pseudoinverse to calculate optimized
      // factor matrix
      //t1 = std::chrono::high_resolution_clock::now();
      gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, pseudoInverse(n, rank, lambda), 0.0, an);
      //t2 = std::chrono::high_resolution_clock::now();
      //time = t2 - t1;
      //final_gemm = time.count();

      // compute the difference between this new factor matrix and the previous
      // iteration
      for (auto l = 0; l < rank; ++l) A[ndim](l) = normCol(an, l);
      auto nrm = norm(A[n] - an);
      if(n == ndim - 2 && symm){
        test += nrm;
      }
      test += nrm;
      A[n] = an;
      //printf("%3.8f\t%3.8f\t%3.8f\t%3.8f", first_gemm, second_gemm, third_gemm, final_gemm);
      //std::cout << std::endl;
    }

    /// Generates V by first Multiply A^T.A then Hadamard product V(i,j) *=
    /// A^T.A(i,j);
    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices

    Tensor generate_V(int n, int rank, double lambda) {
      Tensor V(rank, rank);
      V.fill(1.0);
      auto * V_ptr = V.data();
      for (auto i = 0; i < ndim; ++i) {
        if (i != n) {
          Tensor lhs_prod(rank, rank);
          gemm(CblasTrans, CblasNoTrans, 1.0, A[i], A[i], 0.0, lhs_prod);
          const auto * lhs_ptr = lhs_prod.data();
          for(int j = 0; j < rank*rank; j++)
            *(V_ptr + j) *= *(lhs_ptr +j);
        }
      }

      for(auto j = 0; j < rank; ++j){
        V(j,j) += lambda;
      }
      return V;
    }

    // Keep track of the Left hand Khatri-Rao product of matrices and
    // Continues to multiply be right hand products, skipping
    // the matrix at index n.
    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    /// \param[in] forward Should the Khatri-Rao product move through the factor
    /// matrices in the forward (0 to ndim) or backward (ndim to 0) direction

    Tensor generate_KRP(int n, int rank, bool forward) {
      Tensor temp(Range{Range1{A.at(n).extent(0)}, Range1{rank}});
      Tensor left_side_product(Range{Range1{rank}, Range1{rank}});

      if (forward) {
        for (auto i = 0; i < ndim; ++i) {
          if ((i == 0 && n != 0) || (i == 1 && n == 0)) {
            left_side_product = A.at(i);
          } else if (i != n) {
            khatri_rao_product(left_side_product, A[i], temp);
            left_side_product = temp;
          }
        }
      }

      else {
        for (auto i = ndim - 1; i > -1; --i) {
          if ((i == ndim - 1 && n != ndim - 1) || (i == ndim - 2 && n == ndim - 1)) {
            left_side_product = A.at(i);
          }

          else if (i != n) {
            khatri_rao_product<Tensor>(left_side_product, A[i], temp);
            left_side_product = temp;
          }
        }
      }
      return left_side_product;
    }

    /// \param[in] factor Which factor matrix to normalize
    /// \param[in] col Which column of the factor matrix to normalize
    /// \return The norm of the col column of the factor factor matrix

    double normCol(int factor, int col) {
      const double *AF_ptr = A[factor].data() + col;

      double norm = sqrt(dot(A[factor].extent(0), AF_ptr, A[factor].extent(1), AF_ptr, A[factor].extent(1)));

      scal(A[factor].extent(0), 1 / norm, std::begin(A[factor]) + col, A[factor].extent(1));

      return norm;
    }

    /// \param[in, out] Mat The matrix whose column will be normalized, return
    /// column col normalized matrix.
    /// \param[in] col The column of matrix Mat to be
    /// normalized.
    /// \return the norm of the col column of the matrix Mat

    double normCol(Tensor &Mat, int col) {
      const double *Mat_ptr = Mat.data() + col;

      double norm = sqrt(dot(Mat.extent(0), Mat_ptr, Mat.extent(1), Mat_ptr, Mat.extent(1)));

      scal(Mat.extent(0), 1 / norm, std::begin(Mat) + col, Mat.extent(1));

      return norm;
    }

    /// \param[in] Mat Calculates the 2-norm of the matrix mat
    /// \return the 2-norm.

    double norm(const Tensor &Mat) { return sqrt(dot(Mat, Mat)); }

    /// SVD referencing code from
    /// http://www.netlib.org/lapack/explore-html/de/ddd/lapacke_8h_af31b3cb47f7cc3b9f6541303a2968c9f.html

    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] R The current rank, column dimension of the factor matrices
    /// \return V^{-1} The psuedoinverse of the matrix V.

    Tensor pseudoInverse(int n, int R, double lambda) {
      // CP_ALS method requires the psuedoinverse of matrix V
      auto a = generate_V(n, R, lambda);
      Tensor s(Range{Range1{R}});
      Tensor U(Range{Range1{R}, Range1{R}});
      Tensor Vt(Range{Range1{R}, Range1{R}});

// btas has no generic SVD for MKL LAPACKE
#ifdef _HAS_INTEL_MKL
      double worksize;
      double *work = &worksize;
      lapack_int lwork = -1;
      lapack_int info = 0;

      char A = 'A';

      // Call dgesvd with lwork = -1 to query optimal workspace size:

      info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, A, A, R, R, a.data(), R, s.data(), U.data(), R, Vt.data(), R,
                                 &worksize, lwork);
      if (info)
        ;
      lwork = (lapack_int)worksize;
      work = (double *)malloc(sizeof(double) * lwork);

      info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, A, A, R, R, a.data(), R, s.data(), U.data(), R, Vt.data(), R, work,
                                 lwork);
      if (info)
        ;

      free(work);
#else  // BTAS_HAS_CBLAS

      gesvd('A', 'A', a, s, U, Vt);

#endif

      // Inverse the Singular values with threshold 1e-13 = 0
      double lr_thresh = 1e-13;
      Tensor s_inv(Range{Range1{R}, Range1{R}});
      for (auto i = 0; i < R; ++i) {
        if (s(i) > lr_thresh)
          s_inv(i, i) = 1 / s(i);
        else
          s_inv(i, i) = s(i);
      }
      s.resize(Range{Range1{R}, Range1{R}});

      // Compute the matrix A^-1 from the inverted singular values and the U and
      // V^T provided by the SVD
      gemm(CblasNoTrans, CblasNoTrans, 1.0, U, s_inv, 0.0, s);
      gemm(CblasNoTrans, CblasNoTrans, 1.0, s, Vt, 0.0, U);

      return U;
    }

  };  // class CP_ALS

}  // namespace btas

#endif  // BTAS_GENERIC_CP_RALS_H
