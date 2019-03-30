//
// Created by Karl Pierce on 2/25/19.
//

// TODO reimplement calculate epsilon using FitCheck (Or the same trick)
// Always save a copy of MtKRP to CP

#ifndef BTAS_GENERIC_CP_H
#define BTAS_GENERIC_CP_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <btas/btas.h>
#include <btas/error.h>
#include "core_contract.h"
#include "flatten.h"
#include "khatri_rao_product.h"
#include "randomized.h"
#include "swap.h"
#include "tucker.h"
#include "converge_class.h"
#include "rals_helper.h"

namespace btas{
  namespace detail{
    template<typename T, typename Tensor>
    void set_MtKRP(T& t, Tensor & tensor){
      return;
    }

    template<typename Tensor>
    void set_MtKRP(FitCheck<Tensor> & t, Tensor & tensor){
      t.set_MtKRP(tensor);
    }

    template<typename T>
    void get_fit(T& t, double & epsilon){
      //epsilon = epsilon;
      epsilon = -1;
      return;
    }

    template<typename Tensor>
    void get_fit(FitCheck<Tensor> & t, double & epsilon){
      epsilon = t.get_fit();
      return;
    }
  }//namespace detail

  template <typename Tensor, class ConvClass = NormCheck<Tensor>>
  class CP {
  public:
    CP(int dims) : num_ALS(0) {
#if not defined(BTAS_HAS_CBLAS) || not defined(_HAS_INTEL_MKL)
      BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__, "CP decompositions requires LAPACKE or mkl_lapack");
#endif
#ifdef _HAS_INTEL_MKL
#include <mkl_trans.h>
#endif
      ndim = dims;
    }

    ~CP() = default;

    /// Computes decomposition of the order-N tensor \c tensor
    /// with CP rank = \c rank .
    /// Initial guess for factor matrices start at rank = 1
    /// and build to rank = \c rank by increments of \c step, to minimize
    /// error.

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged
    /// \param[in] step CP_ALS built
    /// from r =1 to r = \c rank. r increments by \c step; default = 1.
    /// \param[in] SVD_initial_guess Should the initial factor matrices be
    /// approximated with left singular values? default = false
    /// \param[in] SVD_rank if \c
    /// SVD_initial_guess is true specify the rank of the initial guess such that
    /// SVD_rank <= rank. default = 0
    /// \param[in] symm is \c tensor is symmetric in the last two dimension? default = false
    /// \param[in]
    /// max_als Max number of iterations allowed to converge the ALS approximation default = 1e4
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = true
    /// \param[in]
    /// calculate_epsilon Should the 2-norm error be calculated \f$ ||T_{\rm exact} -
    /// T_{\rm approx}|| = \epsilon. \f$ Default = false.
    /// \param[in] direct Should the CP decomposition be computed without
    /// calculating the Khatri-Rao product? Default = true.
    /// \returns 2-norm
    /// error between exact and approximate tensor, -1 if calculate_epsilon =
    /// false.

    double compute_rank(int rank, ConvClass &converge_test, int step = 1,
                        bool SVD_initial_guess = false, int SVD_rank = 0, bool symm = false, int max_als = 1e4,
                        bool fast_pI = true, bool calculate_epsilon = false, bool direct = true) {
      if (rank <= 0) BTAS_EXCEPTION("Decomposition rank must be greater than 0");
      if (SVD_initial_guess && SVD_rank > rank) BTAS_EXCEPTION("Initial guess is larger than the desired CP rank");
      double epsilon = -1.0;
      build(rank, converge_test, direct, max_als, calculate_epsilon, step, epsilon, SVD_initial_guess, SVD_rank,
            fast_pI, symm);
      std::cout << "Number of ALS iterations performed: " << num_ALS << std::endl;

      detail::get_fit(converge_test, epsilon);

      return epsilon;
    }

    /// Computes the decomposition of the order-N tensor \c tensor
    /// to \f$ rank \leq \f$ \c max_als such that
    /// \f[ || T_{exact} - T_{approx}||_F = \epsilon \leq tcutCP \f]
    /// with rank incrementing by \c step.

    /// \param[in] converge_test Test to see if ALS is converged
    /// \param[in] tcutCP How small \f$\epsilon\f$ must be to consider the CP
    /// decomposition converged. Default = 1e-2.
    /// \param[in] step CP_ALS built from r =1 to r = \c rank. r
    /// increments by \c step; default = 1.
    /// \param[in] max_rank The highest rank
    /// approximation computed before giving up on CP-ALS. Default = 1e5.
    /// \param[in] SVD_initial_guess Should the initial factor matrices be
    /// approximated with left singular values? default = false
    /// \param[in] SVD_rank if \c
    /// SVD_initial_guess is true specify the rank of the initial guess such that
    /// SVD_rank <= rank. default = true
    /// \param[in] symm is \c tensor is symmetric in the last two dimension? default = false
    /// \param[in] max_als Max number of iterations allowed to converge the ALS
    /// approximation default = 1e4
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = true
    /// \param[in] direct Should the
    /// CP decomposition be computed without calculating the
    /// Khatri-Rao product? Default = true.
    /// \returns 2-norm error
    /// between exact and approximate tensor, \f$ \epsilon \f$
    double compute_error(ConvClass &converge_test, double tcutCP = 1e-2, int step = 1,
                         int max_rank = 1e5, bool SVD_initial_guess = false, int SVD_rank = 0,
                         bool symm = false, double max_als = 1e4, bool fast_pI = true, bool direct = true) {
      int rank = (A.empty()) ? ((SVD_initial_guess) ? SVD_rank : 1) : A[0].extent(0);
      double epsilon = tcutCP + 1;
      while (epsilon > tcutCP && rank < max_rank) {
        build(rank, converge_test, direct, max_als, true, step, epsilon, SVD_initial_guess, SVD_rank, fast_pI, symm);
        rank++;
      }
      detail::get_fit(converge_test, epsilon);
      return epsilon;
    }

    /// Computes decomposition of the order-N tensor \c tensor
    /// with \f$ CP rank \leq \f$ \c desired_rank \n
    /// Initial guess for factor matrices start at rank = 1
    /// and build to rank = \c rank by geometric steps of \c geometric_step, to
    /// minimize error.

    /// \param[in] desired_rank Rank of CP decomposition, r, will build by
    /// geometric step until \f$ r \leq \f$ \c desired_rank.
    /// \param[in] converge_test Test to see if ALS is converged
    /// \param[in] geometric_step CP_ALS built from r =1 to r = \c rank. r increments by r *=
    /// \c geometric_step; default = 2.
    /// \param[in] SVD_initial_guess Should the initial factor matrices be
    /// approximated with left singular values? default = false
    /// \param[in] SVD_rank if \c
    /// SVD_initial_guess is true specify the rank of the initial guess such that
    /// SVD_rank <= rank. default = true
    /// \param[in] symm is \c tensor is symmetric in the last two dimension? default = false
    /// \param[in] max_als Max number of iterations allowed to
    /// converge the ALS approximation. default = 1e4
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
    double compute_geometric(int desired_rank, ConvClass &converge_test, int geometric_step = 2,
                             bool SVD_initial_guess = false, int SVD_rank = 0, bool symm = false, int max_als = 1e4,
                             bool fast_pI = true, bool calculate_epsilon = false, bool direct = true) {
      if (geometric_step <= 0) {
        BTAS_EXCEPTION("The step size must be larger than 0");
      }
      if (SVD_initial_guess && SVD_rank > desired_rank) {
        BTAS_EXCEPTION("Initial guess is larger than desired CP rank");
      }
      double epsilon = -1.0;
      int rank = (SVD_initial_guess) ? SVD_rank : 1;

      while (rank <= desired_rank && rank < max_als) {
        build(rank, converge_test, direct, max_als, calculate_epsilon, geometric_step, epsilon, SVD_initial_guess,
              SVD_rank, fast_pI, symm);
        if (geometric_step <= 1)
          rank++;
        else
          rank *= geometric_step;
      }

      detail::get_fit(converge_test, epsilon);
      return epsilon;
    }

#ifdef _HAS_INTEL_MKL
    virtual double compute_PALS(std::vector<ConvClass> & converge_list, double RankStep = 0.5, int panels = 4, bool symm = false,
                         int max_als = 20,bool fast_pI = true, bool calculate_epsilon = false, bool direct = true) = 0;
#endif // _HAS_INTEL_MKL

    /// returns the rank \c rank optimized factor matrices
    /// \return Factor matrices stored in a vector. For example, a order-3
    /// tensor has factor matrices in positions [0]-[2]. In [3] there is scaling
    /// factor vector of size \c rank
    /// \throw  Exception if the CP decomposition is
    /// not yet computed.

    std::vector<Tensor> get_factor_matrices() {
      if (!A.empty())
        return A;
      else BTAS_EXCEPTION("Attempting to return a NULL object. Compute CP decomposition first.");
    }

    /// Function that uses the factor matrices from the CP
    /// decomposition and reconstructs the
    /// approximated tensor
    /// \returns The tensor approxmimated from the factor
    /// matrices of the CP decomposition.
    /// \throws Exception if the CP decomposition is
    /// not yet computed.
    virtual Tensor reconstruct() {
      if (A.empty()) BTAS_EXCEPTION(
              "Factor matrices have not been computed. You must first calculate CP decomposition.");

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
        scal(A[0].extent(0), 1 / A[ndim](i), std::begin(A[0]) + i, rank);
      }
      return hold;
    }
    Tensor reconstruct(std::vector<int> dims){
      std::vector<size_t> dimensions;
      for (int i = 0; i < ndim; i++) {
        dimensions.push_back(A[dims[i]].extent(0));
      }
      auto rank = A[0].extent(1);
      for (int i = 0; i < rank; i++) {
        scal(A[dims[0]].extent(0), A[ndim](i), std::begin(A[dims[0]]) + i, rank);
      }

      // Make the Khatri-Rao product of all the factor matrices execpt the last dimension
      Tensor KRP = A[dims[0]];
      Tensor hold = A[dims[0]];
      for (int i = 1; i < A.size() - 2; i++) {
        khatri_rao_product(KRP, A[dims[i]], hold);
        KRP = hold;
      }

      // contract the rank dimension of the Khatri-Rao product with the rank dimension of
      // the last factor matrix. hold is now the reconstructed tensor
      hold = Tensor(KRP.extent(0), A[dims[ndim - 1]].extent(0));
      gemm(CblasNoTrans, CblasTrans, 1.0, KRP, A[dims[ndim - 1]], 0.0, hold);

      // resize the reconstructed tensor to the correct dimensions
      hold.resize(dimensions);

      // remove the scaling applied to the first factor matrix
      for (int i = 0; i < rank; i++) {
        scal(A[dims[0]].extent(0), 1 / A[ndim](i), std::begin(A[dims[0]]) + i, rank);
      }
      return hold;
    }
    void print(Tensor& tensor){
      if(tensor.rank() == 2){
        int row = tensor.extent(0), col = tensor.extent(1);
        for(int i = 0; i < row; ++i){
          const auto * tensor_ptr = tensor.data() + i * col;
          for(int j = 0; j < col; ++j){
            //os << *(tensor_ptr + j) << ",\t";
            std::cout << *(tensor_ptr + j) << ",\t";
          }
          std::cout << std::endl;
        }
      }
      else{
        for(auto &i: tensor){
          //os << i << ", \t";
          std::cout << i << ",";
        }
      }
      std::cout << std::endl;
      return ;
    }

  protected:
    int num_ALS;
    std::vector<Tensor> A;
    int ndim;

    virtual void build(int rank, ConvClass &converge_test, bool direct, int max_als, bool calculate_epsilon, int step, double &epsilon,
                  bool SVD_initial_guess, int SVD_rank, bool & fast_pI, bool symm) = 0;

    /// Generates V by first Multiply A^T.A then Hadamard product V(i,j) *=
    /// A^T.A(i,j);
    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    Tensor generate_V(int n, int rank, double lambda = 0.0) {
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

    Tensor normCol(int factor) {
      if(factor >= ndim) BTAS_EXCEPTION("Factor is out of range");
      auto rank = A[factor].extent(1);
      auto size = A[factor].size();
      Tensor lambda(rank);
      lambda.fill(0.0);
      auto A_ptr = A[factor].data();
      auto lam_ptr = lambda.data();
      for(int i = 0; i < size; ++i){
        *(lam_ptr + i % rank) += *(A_ptr + i) * *(A_ptr + i);
      }
      for(int i = 0; i < rank; ++i){
        *(lam_ptr + i) = sqrt(*(lam_ptr + i));
      }
      for(int i = 0; i < size; ++i){
        *(A_ptr + i) /= *(lam_ptr + i % rank);
      }
      return lambda;
    }

    /// \param[in, out] Mat The matrix whose column will be normalized, return
    /// column col normalized matrix.
    /// \param[in] col The column of matrix Mat to be
    /// normalized.
    /// \return the norm of the col column of the matrix Mat

    void normCol(Tensor &Mat) {
      if(Mat.rank() > 2) BTAS_EXCEPTION("normCol with rank > 2 not yet supported");
      auto rank = Mat.extent(1);
      auto size = Mat.size();
      A[ndim].fill(0.0);
      auto Mat_ptr = Mat.data();
      auto A_ptr = A[ndim].data();
      for(int i = 0; i < size; ++i){
        *(A_ptr + i % rank) += *(Mat_ptr + i) * *(Mat_ptr + i);
      }
      for(int i = 0; i < rank; ++i){
        *(A_ptr + i) = sqrt(*(A_ptr + i));
      }
      for(int i = 0; i < size; ++i){
        *(Mat_ptr + i) /= *(A_ptr + i % rank);
      }
    }

    /// \param[in] Mat Calculates the 2-norm of the matrix mat
    /// \return the 2-norm.

    double norm(const Tensor &Mat) { return sqrt(dot(Mat, Mat)); }

    /// SVD referencing code from
    /// http://www.netlib.org/lapack/explore-html/de/ddd/lapacke_8h_af31b3cb47f7cc3b9f6541303a2968c9f.html
    /// Fast pseudo-inverse algorithm described in
    /// https://arxiv.org/pdf/0804.4809.pdf

    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] R The current rank, column dimension of the factor matrices
    /// \param[in] symm does the reference tensor have symmetry in the last two modes
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// \return V^{\dagger} The psuedoinverse of the matrix V.

    Tensor pseudoInverse(int n, int R, bool & fast_pI, double lambda = 0.0) {
      // CP_ALS method requires the psuedoinverse of matrix V
#ifdef _HAS_INTEL_MKL
      if(fast_pI) {
        auto a = this->generate_V(n, R, lambda);
        Tensor temp(R, R), inv(R, R);
        // V^{\dag} = (A^T A) ^{-1} A^T
        gemm(CblasTrans, CblasNoTrans, 1.0, a, a, 0.0, temp);
        fast_pI = Inverse_Matrix(temp);
        if(fast_pI) {
          gemm(CblasNoTrans, CblasTrans, 1.0, temp, a, 0.0, inv);
          return inv;
        }
        else{
          std::cout << "Fast pseudo-inverse failed reverting to normal pseudo-inverse" << std::endl;
        }
      }
#else
      fast_pI = false;
#endif // _HAS_INTEL_MKL

      if(!fast_pI) {
        auto a = this->generate_V(n, R, lambda);
        Tensor s(Range{Range1{R}});
        Tensor U(Range{Range1{R}, Range1{R}});
        Tensor Vt(Range{Range1{R}, Range1{R}});

// btas has no generic SVD for MKL LAPACKE
//        time1 = std::chrono::high_resolution_clock::now();
#ifdef _HAS_INTEL_MKL
        double worksize;
        double *work = &worksize;
        lapack_int lwork = -1;
        lapack_int info = 0;

        char A = 'A';

        // Call dgesvd with lwork = -1 to query optimal workspace size:

        info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, A, A, R, R, a.data(), R, s.data(), U.data(), R, Vt.data(), R,
                                   &worksize, lwork);
        if (info != 0)
          BTAS_EXCEPTION("SVD pseudo inverse failed");

        lwork = (lapack_int) worksize;
        work = (double *) malloc(sizeof(double) * lwork);

        info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, A, A, R, R, a.data(), R, s.data(), U.data(), R, Vt.data(), R, work,
                                   lwork);
        if (info != 0)
          BTAS_EXCEPTION("SVD pseudo inverse failed");

        free(work);
#else  // BTAS_HAS_CBLAS

        gesvd('A', 'A', a, s, U, Vt);

#endif

        // Inverse the Singular values with threshold 1e-13 = 0
        double lr_thresh = 1e-13;
        Tensor s_inv(Range{Range1{R}, Range1{R}});
        s_inv.fill(0.0);
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
      else{
        BTAS_EXCEPTION("Pseudo inverse failed" );
      }
    }

    Tensor pseudoInverse(Tensor & a){
      bool matlab = false;
      auto R = A[0].extent(1);
      Tensor s(Range{Range1{R}});
      Tensor U(Range{Range1{R}, Range1{R}});
      Tensor Vt(Range{Range1{R}, Range1{R}});

      if(! matlab) {

// btas has no generic SVD for MKL LAPACKE
//        time1 = std::chrono::high_resolution_clock::now();
#ifdef _HAS_INTEL_MKL
        double worksize;
        double *work = &worksize;
        lapack_int lwork = -1;
        lapack_int info = 0;

        char A = 'A';

        // Call dgesvd with lwork = -1 to query optimal workspace size:

        info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, A, A, R, R, a.data(), R, s.data(), U.data(), R, Vt.data(), R,
                                   &worksize, lwork);
        if (info != 0)
        BTAS_EXCEPTION("SVD pseudo inverse failed");

        lwork = (lapack_int) worksize;
        work = (double *) malloc(sizeof(double) * lwork);

        info = LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, A, A, R, R, a.data(), R, s.data(), U.data(), R, Vt.data(), R, work,
                                   lwork);
        if (info != 0)
        BTAS_EXCEPTION("SVD pseudo inverse failed");

        free(work);
#else  // BTAS_HAS_CBLAS

        gesvd('A', 'A', a, s, U, Vt);

#endif

        // Inverse the Singular values with threshold 1e-13 = 0
        double lr_thresh = 1e-13;
        Tensor s_inv(Range{Range1{R}, Range1{R}});
        s_inv.fill(0.0);
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

      }
      return U;
    }
  };

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
    A.compute_rank(rank, converge_test)             // Computes the CP_ALS of tensor to
                                                    // rank.

    A.compute_error(converge_test, omega)           // Computes the CP_ALS of tensor to
                                                    // 2-norm
                                                    // error < omega.

    A.compute_geometric(rank, converge_test, step)  // Computes CP_ALS of tensor to
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
  template <typename Tensor, class ConvClass = NormCheck<Tensor>>
  class CP_ALS : public CP<Tensor, ConvClass>{
  public:
      using CP<Tensor,ConvClass>::A;
      using CP<Tensor,ConvClass>::ndim;
    CP_ALS(Tensor& tensor): CP<Tensor,ConvClass>(tensor.rank()), tensor_ref(tensor),
    size(tensor.size()){

    }

    ~CP_ALS() = default;

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
      if(converge_list.size() < panels) BTAS_EXCEPTION("Too few convergence tests.  Must provide a list of panels convergence tests");
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
          this->build(max_dim, converge_test, direct, max_als, calculate_epsilon, 1, epsilon, true, max_dim, fast_pI, symm);
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
              std::mt19937 generator(3);
              std::normal_distribution<double> distribution(0, 2);
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
            this->normCol(0);
          }
          ALS(rank_new, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
        }
        count++;
        //if (count + 1 == panels) max_als = 1000;
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
    /// \param[in] step CP_ALS built
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
                                   double max_als = 1e4, double ALStol = 0.1, bool SVD_initial_guess = false,
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
    /// CP_ALS built from r =1 to r = rank. r increments by step; default = 1.
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
    Tensor& tensor_ref;
    int size;
    bool factors_set = false;

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
               bool SVD_initial_guess, int SVD_rank, bool & fast_pI, bool symm) override {
      // If its the first time into build and SVD_initial_guess
      // build and optimize the initial guess based on the left
      // singular vectors of the reference tensor.
#ifdef _HAS_INTEL_MKL
      if (A.empty() && SVD_initial_guess) {
        if (SVD_rank == 0) BTAS_EXCEPTION("Must specify the rank of the initial approximation using SVD");

        std::vector<int> modes_w_dim_LT_svd;
        A = std::vector<Tensor>(ndim);

        // Determine which factor matrices one can fill using SVD initial guess
        for(int i = 0; i < ((symm) ? ndim - 1: ndim); i++){
          if(tensor_ref.extent(i) < SVD_rank){
            modes_w_dim_LT_svd.push_back(i);
          }
        }

        // Fill all factor matrices with their singular vectors,
        // because we contract X X^T (where X is reference tensor) to make finding
        // singular vectors an eigenvalue problem some factor matrices will not be
        // full rank;
        A[0] = Tensor(tensor_ref.extent(0), SVD_rank);
        A[0].fill(0.0);

        for(int i = 1; i < ((symm) ? ndim - 1: ndim); i++){
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
        std::mt19937 generator(3);
        // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
        for(auto& i: modes_w_dim_LT_svd){
          int R = tensor_ref.extent(i);
          auto lower_bound = {0, R};
          auto upper_bound = {R, SVD_rank};
          auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
          std::normal_distribution<double> distribution(0, 2.0);
          for(auto iter = view.begin(); iter != view.end(); ++iter){
            *(iter) = distribution(generator);
          }
        }

        // Normalize the columns of the factor matrices and
        // set the values al lambda, the weigt of each order 1 tensor
        Tensor lambda(Range{Range1{SVD_rank}});
        A.push_back(lambda);
        for(auto i = 0; i < ((symm) ? ndim - 1: ndim); ++i){
          this->normCol(A[i]);
        }

        // Optimize this initial guess.
        ALS(SVD_rank, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
      }
#else  // _HAS_INTEL_MKL
      if (SVD_initial_guess) BTAS_EXCEPTION("Computing the SVD requires LAPACK");
#endif // _HAS_INTEL_MKL
      // This loop keeps track of column dimension
      bool opt_in_for_loop = false;
      for (auto i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += step) {
        opt_in_for_loop = true;
        // This loop walks through the factor matrices
        for (auto j = 0; j < ndim; ++j) {  // select a factor matrix
          // If no factor matrices exists, make a set of factor matrices
          // and fill them with random numbers that are column normalized
          // and create the weighting vector lambda
          if (i == 0) {
            Tensor a(Range{tensor_ref.range(j), Range1{i + 1}});
            a.fill(rand());
            A.push_back(a);
            this->normCol(j);
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
              std::mt19937 generator(3);
              std::normal_distribution<double> distribution(0, 2);
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
        // compute the ALS of factor matrices with rank = i + 1.
        ALS(i + 1, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
      }
      if(factors_set && ! opt_in_for_loop){
        ALS(rank, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
      }
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

      if(symm){
        A[ndim - 1] = A[ndim -2];
      }

      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      //std::cout << "count\tfit\tFit Change" << std::endl;
      while(count < max_als && !is_converged){
        count++;
        this->num_ALS++;
        for (auto i = 0; i < ((symm) ? ndim - 1: ndim); i++) {
          if (dir)
            direct(i, rank, symm, fast_pI, matlab, converge_test);
          else
            update_w_KRP(i, rank, symm, fast_pI, converge_test);
        }
        if(symm){
          A[ndim - 1] = A[ndim - 2];
        }
        //std::cout << count << "\t";
        is_converged = converge_test(A);
        //T *= 0.6;
      }

      // Checks loss function if required
      if (calculate_epsilon) {
        if (typeid(converge_test) == typeid(btas::FitCheck<Tensor>)) {
          detail::get_fit(converge_test, epsilon);
          epsilon = 1 - epsilon;
        } else{
          epsilon = this->norm(this->reconstruct() - tensor_ref);
        }
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
    void update_w_KRP(int n, int rank, bool symm, bool & fast_pI, ConvClass & converge_test) {
      Tensor temp(A[n].extent(0), rank);
      Tensor an(A[n].range());

#ifdef _HAS_INTEL_MKL

      // Computes the Khatri-Rao product intermediate
      auto KhatriRao = this->generate_KRP(n, rank, true);

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
      gemm(CblasNoTrans, CblasNoTrans, 1.0, flatten(tensor_ref, n), this->generate_KRP(n, rank, true), 0.0, temp);
#endif

      detail::set_MtKRP(converge_test, temp);
//      if(typeid(converge_test) == typeid(btas::FitCheck<Tensor>)) {
//        converge_test.set_MtKRP(temp);
//      }
      // contract the product from above with the psuedoinverse of the Hadamard
      // produce an optimize factor matrix
      gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, this->pseudoInverse(n, rank, fast_pI), 0.0, an);

      // compute the difference between this new factor matrix and the previous
      // iteration
      //for (auto l = 0; l < rank; ++l) A[ndim](l) = normCol(an, l);
      this->normCol(an);

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

    void direct(int n, int rank, bool symm, bool & fast_pI, bool & matlab, ConvClass & converge_test) {

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
      // multiply resulting matrix temp by pseudoinverse to calculate optimized
      // factor matrix

      detail::set_MtKRP(converge_test, temp);
//      if(typeid(converge_test) == typeid(btas::FitCheck<Tensor>)) {
//        converge_test.set_MtKRP(temp);
//      }
      //t1 = std::chrono::high_resolution_clock::now();
#ifdef _HAS_INTEL_MKL
      if(fast_pI && matlab) {
        // This method computes the inverse quickly for a square matrix
        // based on MATLAB's implementation of A / B operator.
        btas::Tensor<int, DEFAULT::range, varray<int> > piv(rank);
        piv.fill(0);

        auto a = this->generate_V(n, rank);
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
        gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, this->pseudoInverse(n, rank, fast_pI), 0.0, an);
      }
#else
      matlab = false;
      if(!fast_pI || !matlab){
        gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, this->pseudoInverse(n, rank, fast_pI), 0.0, an);
      }
#endif
      //t2 = std::chrono::high_resolution_clock::now();
      //time = t2 - t1;
      //gemm_wPI += time.count();


      // Normalize the columns of the new factor matrix and update
      this->normCol(an);
      //Simulated Annealing
      /*{
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> dist(0.0, T);
        for(auto & i: an){
          i += dist(generator);
        }
      }*/
      A[n] = an;
      //this->normCol(n);
    }

  };

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
  template <typename Tensor, class ConvClass = NormCheck<Tensor>>
  class CP_RALS : public CP<Tensor, ConvClass>{

    using CP<Tensor,ConvClass>::A;
    using CP<Tensor,ConvClass>::ndim;
    using CP<Tensor,ConvClass>::pseudoInverse;
    using CP<Tensor,ConvClass>::normCol;
    using CP<Tensor,ConvClass>::generate_KRP;
    using CP<Tensor,ConvClass>::generate_V;
    using CP<Tensor,ConvClass>::norm;

  public:
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
              std::mt19937 generator(3);
              std::normal_distribution<double> distribution(0, 2);
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
                                   double max_als = 1e4, double ALStol = 0.1, bool SVD_initial_guess = false,
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
        std::mt19937 generator(3);
        // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
        for(auto& i: modes_w_dim_LT_svd){
          int R = tensor_ref.extent(i);
          auto lower_bound = {0, R};
          auto upper_bound = {R, SVD_rank};
          auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
          std::normal_distribution<double> distribution(0, 2.0);
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
              std::mt19937 generator(3);
              std::normal_distribution<double> distribution(0, 2);
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
    A.compute_rank(rank, converge_test)             // Computes the CP_ALS of tensor to
                                                    // rank.

    A.compute_error(converge_test, omega)           // Computes the CP_ALS of tensor to
                                                    // 2-norm
                                                    // error < omega.

    A.compute_geometric(rank, converge_test, step)  // Computes CP_ALS of tensor to
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
  template <typename Tensor, class ConvClass = NormCheck<Tensor>>
  class CP_DF_ALS : public CP<Tensor, ConvClass> {

    using CP<Tensor,ConvClass>::A;
    using CP<Tensor,ConvClass>::ndim;
    using CP<Tensor,ConvClass>::pseudoInverse;
    using CP<Tensor,ConvClass>::normCol;
    using CP<Tensor,ConvClass>::generate_KRP;
    using CP<Tensor,ConvClass>::generate_V;
    using CP<Tensor,ConvClass>::norm;

  public:
    /// Constructor of object CP_ALS
    /// \param[in] tensor The tensor object to be decomposed
    CP_DF_ALS(Tensor &left, Tensor &right, std::vector<int> &symm) :
            CP<Tensor,ConvClass>(left.rank() + right.rank() - 2),
            tensor_ref_left(left), tensor_ref_right(right),
            ndimL(left.rank()), ndimR(right.rank()),
            symm_dims(symm)
            {
      if (symm_dims.size() < ndim) BTAS_EXCEPTION(
              "Tensor describing symmetries must be equal to number of non-connectin dimensions");
    }

    CP_DF_ALS(Tensor &left, Tensor &right) :
    CP<Tensor,ConvClass>(left.rank() + right.rank() - 2),
    tensor_ref_left(left), tensor_ref_right(right),
    ndimL(left.rank()), ndimR(right.rank()) {
      for(int i = 0; i < ndim; ++i){
        symm_dims.push_back(i);
      }
    }

    CP_DF_ALS() = default;

    ~CP_DF_ALS() = default;

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
    /// \param[in] max_als Max number of iterations allowed to
    /// converge the ALS approximation. default = 20
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = true
    /// \param[in] calculate_epsilon Should the
    /// 2-norm error be calculated \f$ ||T_{exact} - T_{approx}|| = \epsilon \f$.
    /// Default = false.

    /// \returns 2-norm error
    /// between exact and approximate tensor, -1.0 if calculate_epsilon = false,
    /// \f$ \epsilon \f$
    double compute_PALS(std::vector<ConvClass> & converge_list, double RankStep = 0.5, int panels = 4, bool symm = false,
                        int max_als = 20,bool fast_pI = true, bool calculate_epsilon = false, bool direct = true) override{
      if (RankStep <= 0) BTAS_EXCEPTION("Panel step size cannot be less than or equal to zero");
      if(converge_list.size() < panels) BTAS_EXCEPTION("Too few convergence tests.  Must provide a list of panels convergence tests");
      double epsilon = -1.0;
      int count = 0;
      // Find the largest rank this will be the first panel
      auto max_dim = tensor_ref_left.extent(0);
      for(int i = 1; i < ndimL; ++i){
        auto dim = tensor_ref_left.extent(i);
        max_dim = ( dim > max_dim ? dim : max_dim);
      }
      for(int i = 0; i < ndimR; ++i){
        auto dim = tensor_ref_right.extent(i);
        max_dim = (dim > max_dim ? dim : max_dim);
      }

      while(count < panels){
        auto converge_test = converge_list[count];
        // Use tucker initial guess (SVD) to compute the first panel
        if(count == 0) {
          build(max_dim, converge_test, direct, max_als, calculate_epsilon, 1, epsilon, true, max_dim, fast_pI, symm);
          //build(max_dim, converge_test, max_als, calculate_epsilon, 1, epsilon, true, max_dim, fast_pI);
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
              std::mt19937 generator(3);
              std::normal_distribution<double> distribution(0, 2);
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
          ALS(rank_new, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
        count++;
        //if (count + 1 == panels) max_als = 1000;
      }
      std::cout << "Number of ALS iterations was " << this->num_ALS << std::endl;
      return epsilon;
    }
#endif // _HAS_INTEL_MKL

  protected:
    Tensor &tensor_ref_left;
    Tensor &tensor_ref_right;
    int ndimL;
    int ndimR;
    std::vector<int> symm_dims;

    void build(int rank, ConvClass &converge_test, bool direct, int max_als, bool calculate_epsilon, int step, double &epsilon,
               bool SVD_initial_guess, int SVD_rank, bool & fast_pI, bool symm) override{
      {
        bool factors_set = false;
        // If its the first time into build and SVD_initial_guess
        // build and optimize the initial guess based on the left
        // singular vectors of the reference tensor.
#ifdef _HAS_INTEL_MKL
        if (A.empty() && SVD_initial_guess) {
          if (SVD_rank == 0) BTAS_EXCEPTION("Must specify the rank of the initial approximation using SVD");

          // easier to do this part by constructing tensor_ref
          // This is an N^5 step but it is only done once so it shouldn't be that expensive.
          // once the code is working this step can be reduced to N^4 (though it might be less accurate that way)
          // it is something to test.
          // Must reconstruct with matrix multiplication so
          // get size of product of dimensions (not connecting) for left and right side
          // Also need tensor dimensions to resize tensor_ref after contraction
          std::vector<int> TRdims(ndim);
          auto trLsize = 1.0, trRsize = 1.0;
          for(auto i = 1; i < ndimL; ++i){
            auto dim = tensor_ref_left.extent(i);
            TRdims[i - 1] = dim;
            trLsize *= dim;
          }
          for(auto i = 1; i < ndimR; ++i){
            auto dim = tensor_ref_right.extent(i);
            // i = 1 must take it to 0; then add left dimensions; then subtract 1 for connecting dimension
            TRdims[i + ndimR - 2] = dim;
            trRsize *= dim;
          }

          // Make TR with correct L/R size
          Tensor tensor_ref(trLsize, trRsize);

          // save ranges for after contraction to resize
          auto TRLrange = tensor_ref_left.range();
          auto TRRrange = tensor_ref_right.range();

          // resize tensors to matrices
          tensor_ref_left.resize(Range{Range1{tensor_ref_left.extent(0)},Range1{trLsize}});
          tensor_ref_right.resize(Range{Range1{tensor_ref_right.extent(0)},Range1{trRsize}});

          // matrix multiplication
          gemm(CblasTrans, CblasNoTrans, 1.0, tensor_ref_left, tensor_ref_right, 0.0, tensor_ref);

          // Resize tensor_ref's back to original dimensions
          tensor_ref_left.resize(TRLrange);
          tensor_ref_right.resize(TRRrange);
          tensor_ref.resize(TRdims);

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
          A[0] = Tensor(tensor_ref.extent(0), SVD_rank);
          A[0].fill(0.0);

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
          std::mt19937 generator(3);
          // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
          for(auto& i: modes_w_dim_LT_svd){
            int R = tensor_ref.extent(i);
            auto lower_bound = {0, R};
            auto upper_bound = {R, SVD_rank};
            auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
            std::normal_distribution<double> distribution(0, 2.0);
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

          // Optimize this initial guess.
          ALS(SVD_rank, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
#else  // _HAS_INTEL_MKL
        if (SVD_initial_guess) BTAS_EXCEPTION("Computing the SVD requires LAPACK");
#endif // _HAS_INTEL_MKL
        // This loop keeps track of column dimension
        bool opt_in_for_loop = false;
        for (auto i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += step) {
          opt_in_for_loop = true;
          // This loop walks through the factor matrices
          for (auto j = 0; j < ndim; ++j) {  // select a factor matrix
            // If no factor matrices exists, make a set of factor matrices
            // and fill them with random numbers that are column normalized
            // and create the weighting vector lambda
            if (i == 0) {
              Tensor a;
              if(j < ndimL - 1){
                a = Tensor(Range{Range1{tensor_ref_left.range(j+1)}, Range1{i + 1}});
              }
              else{
                a = Tensor(Range{Range1{tensor_ref_right.range(j - ndimL + 2)}, Range1{i + 1}});
              }
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
                std::mt19937 generator(3);
                std::normal_distribution<double> distribution(0, 2);
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
          // compute the ALS of factor matrices with rank = i + 1.
          ALS(i + 1, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
        if(factors_set && ! opt_in_for_loop){
          ALS(rank, converge_test, max_als, calculate_epsilon, epsilon, fast_pI);
        }
      }
    }

    void ALS(int rank, ConvClass & converge_test, int max_als, bool calculate_epsilon, double &epsilon, bool & fast_pI) {
      auto count = 0;
      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      Tensor MtKRP(A[ndim - 1].extent(0), rank);
      std::cout << "count\tfit\tchange" << std::endl;
      while(count < max_als && !is_converged){
        count++;
        this->num_ALS++;
        for (auto i = 0; i < ndim; i++) {
          auto tmp = symm_dims[i];
          if(tmp == i) {
            direct(i, rank, matlab, converge_test);
          }
          else if(tmp < i){
            //std::cout << "Skiping the " << i << "th dimension" << std::endl;
            A[i] = A[tmp];
          }
          else{
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

    void direct(int n, int rank, bool & matlab, ConvClass & converge_test) {

      // Determine if n is in the left or the right tensor
      bool leftTensor = n < (ndimL - 1);

      // form the intermediate tensor K
      Tensor K(tensor_ref_left.extent(0), rank);
      // works for leftTensor=true
      //
      {
        // want the tensor without n if n is in the left tensor take the right one and vice versa
        auto & tensor_ref = leftTensor ? tensor_ref_right : tensor_ref_left;

        // How many dimension in this side of the tensor
        auto ndimCurr = tensor_ref.rank();
        auto sizeCurr = tensor_ref.size();
        std::cout << "tensor_ref: " << std::endl;
        for(int i = 0; i < ndimCurr; ++i){
          std::cout << tensor_ref.extent(i) << ",";
        }
        std::cout << std::endl;
        // save range for resize at the end.
        auto R = tensor_ref.range();

        // Start by contracting with the last dimension of tensor without n
        // This is important for picking the correct factor matrix
        // not for picking from tensor_ref
        int contract_dim_inter = leftTensor ? ndim - 1 : ndimL - 2;

        // This is for size of the dimension being contracted
        // picked from tensor_ref
        auto contract_size = tensor_ref.extent(ndimCurr -1);

        // Make the intermediate that will be contracted then hadamard contracted
        // Also resize the tensor for gemm contraction
        Tensor contract_tensor(sizeCurr / contract_size, rank);
        tensor_ref.resize(Range{Range1{sizeCurr / contract_size}, Range1{contract_size}});

        // Contract out the last dimension
        gemm(CblasNoTrans, CblasNoTrans, 1.0, tensor_ref, A[contract_dim_inter], 0.0, contract_tensor);
        // Resize tensor_ref back to original size
        tensor_ref.resize(R);

        // This is the size of the LH dimension of contract_tensor
        sizeCurr /= tensor_ref.extent(ndimCurr - 1);
        // for A now choose the next factor matrix
        --contract_dim_inter;

        // Now want to hadamard contract all the dimension that aren't the connecting dimension
        for(int i = 0; i < ndimCurr - 2; ++i, --contract_dim_inter){
          // The contract_size now starts at the second last dimension
          contract_size = tensor_ref.extent(ndimCurr - 2 - i);
          // Store the LH most dimension size in idx1
          auto idx1 = sizeCurr / contract_size;
          contract_tensor.resize(Range{Range1{idx1},Range1{contract_size},Range1{rank}});
          // After hadamard product middle dimension is gone
          Tensor temp(idx1, rank);

          for(int j = 0; j < idx1; ++j){
            //auto * temp_ptr = temp.data() + j * rank;
            for(int k = 0; k < contract_size; ++k){
              //const auto * contract_ptr = contract_tensor.data() + j * contract_size * rank + k * rank;
              //const auto * A_ptr = A[contract_dim_inter].data() + k * rank;
              for(int r = 0; r < rank; ++r){
                //*(temp_ptr + r) += *(contract_ptr + r) * *(A_ptr + r);
                temp(j,r) += contract_tensor(j,k,r) * A[contract_dim_inter](k,r);
              }
            }
          }
          // After hadamard contract reset contract_tensor with new product
          contract_tensor = temp;
          // Remove the contracted dimension from the current size.
          sizeCurr = idx1;
        }

        // set the hadamard contracted tensor to the intermediate K
        K = contract_tensor;
      }

      // contract K with the other side tensor
      // Tensor_ref now can be the side that contains n
      Tensor & tensor_ref = leftTensor ? tensor_ref_left : tensor_ref_right;
      // Modifying the dimension of tensor_ref so store the range here to resize
      // after contraction.
      Range R = tensor_ref.range();
      // make the new factor matrix for after process
      Tensor an(A[n].range());

      // LH side of tensor after contracting (doesn't include rank or connecting dimension)
      auto LH_size = tensor_ref.size() / tensor_ref.extent(0);
      // Temp holds the intermediate after contracting out the connecting dimension
      // It will be set up to enter hadamard product loop
      Tensor contract_tensor = Tensor(LH_size, rank);
      // resize tensor_ref to remove connecting dimension
      tensor_ref.resize(Range{Range1{tensor_ref.extent(0)},Range1{LH_size}});

      gemm(CblasTrans, CblasNoTrans, 1.0, tensor_ref, K, 0.0, contract_tensor);
      // resize tensor_ref back to original dimensions
      tensor_ref.resize(R);

      std::vector<int> dims(tensor_ref.rank());
      for(int i = 1; i < tensor_ref.rank(); ++i){
        dims[i - 1] = tensor_ref.extent(i);
      }
      dims[dims.size() - 1] = rank;

      // If hadamard loop has to skip a dimension it is stored here.
      auto pseudo_rank = rank;
      // number of dimensions in tensor_ref
      auto ndimCurr = tensor_ref.rank();
      // the dimension that is being hadamard contracted out.
      auto contract_dim = ndimCurr - 2;
      auto nInTensor = leftTensor ? n : n - ndimL + 1;
      auto a_dim = leftTensor ? contract_dim : ndim - 1;
      auto offset = 0;

      // TODO fix the hadamard contraction loop
      // go through hadamard contract on all dimensions excluding rank (will skip one dimension)
      for(int i = 0; i < ndimCurr - 2; ++i, --contract_dim, --a_dim){
        auto contract_size = dims[contract_dim];
        auto idx1 = LH_size / contract_size;
        contract_tensor.resize(Range{Range1{idx1}, Range1{contract_size}, Range1{pseudo_rank}});
        Tensor temp(Range{Range1{idx1}, Range1{pseudo_rank}});

        temp.fill(0.0);

        // If the middle dimension is the mode not being contracted, I will move
        // it to the right hand side temp((size of tensor_ref/product of
        // dimension contracted, rank * mode n dimension)
        if(nInTensor == contract_dim){
          pseudo_rank *= contract_size;
          offset = contract_size;
        }

          // If the code hasn't hit the mode of interest yet, it will contract
          // over the middle dimension and sum over the rank.
        else if (contract_dim > nInTensor){
          for(int j = 0; j < idx1; ++j){
            //auto * temp_ptr = temp.data() + j * pseudo_rank;
            for(int k = 0; k < contract_size; ++k){
              //const auto * contract_ptr = contract_tensor.data() + j * contract_size * pseudo_rank + k * pseudo_rank;
              //const auto * A_ptr = A[a_dim].data() + k * rank;
              for(int r = 0; r < rank; ++r){
                //*(temp_ptr + r) += *(contract_ptr + r) + *(A_ptr + r);
                temp(j,r) += contract_tensor(j,k,r) * A[a_dim](k,r);
              }
            }
          }
          contract_tensor = temp;
        }
          // If the code has passed the mode of interest, it will contract over
          // the middle dimension and sum over rank * mode n dimension
        else{
          for(int j = 0; j < idx1; ++j){
            //auto * temp_ptr = temp.data() + j * pseudo_rank;
            for(int k = 0; k < contract_size; ++k){
              //const auto * contract_ptr = contract_tensor.data() + j * contract_size * pseudo_rank + k * pseudo_rank;
              //const auto * A_ptr = A[a_dim].data() + k * rank;
              for(int l = 0; l < offset; ++l){
                for(int r = 0; r < rank; ++r){
                  //*(temp_ptr + l * rank + r) += *(contract_ptr + l * rank + r) * *(A_ptr + r);
                  temp(j, l*rank + r) += contract_tensor(j,k,l*rank+r) * A[a_dim](k,l*rank+r);
                }
              }
            }
          }

          contract_tensor = temp;
        }
      }

      // If the mode of interest is the 0th mode, then the while loop above
      // contracts over all other dimensions and resulting temp is of the
      // correct dimension If the mode of interest isn't 0th mode, must contract
      // out the 0th mode here, the above algorithm can't perform this
      // contraction because the mode of interest is coupled with the rank
      if(nInTensor != 0){
        auto contract_size = contract_tensor.extent(0);
        Tensor temp(Range{Range1{offset}, Range1{rank}});
        temp.fill(0.0);

        for(int i = 0; i < contract_size; i++){
          //const auto * A_ptr = A[a_dim].data() + i * rank;
          for(int j = 0; j < offset; j++){
            //const auto * contract_ptr = contract_tensor.data() + i * offset * rank + j * rank;
            //auto * temp_ptr = temp.data() + j * rank;
            for(int r = 0; r < rank; r++){
              //*(temp_ptr + r) += *(A_ptr + r) * *(contract_ptr + r);
              temp(j,r) += contract_tensor(i,j,r) * A[a_dim](i,r);
            }
          }
        }

        contract_tensor = temp;
      }

      detail::set_MtKRP(converge_test, contract_tensor);
      // multiply resulting matrix temp by pseudoinverse to calculate optimized
      // factor matrix
      //t1 = std::chrono::high_resolution_clock::now();
#ifdef _HAS_INTEL_MKL
      if(matlab) {
        // This method computes the inverse quickly for a square matrix
        // based on MATLAB's implementation of A / B operator.
        btas::Tensor<int, DEFAULT::range, varray<int> > piv(rank);
        piv.fill(0);

        auto a = generate_V(n, rank);
        int LDB = contract_tensor.extent(0);
        auto info = LAPACKE_dgesv(CblasColMajor, rank, LDB, a.data(), rank, piv.data(), contract_tensor.data(), rank);
        if (info == 0) {
          an = contract_tensor;
        }
        else{
          // If inverse fails resort to the pseudoinverse
          std::cout << "Matlab square inverse failed revert to fast inverse" << std::endl;
          matlab = false;
        }
      }
      if(! matlab){
        bool fast = false;
        if(ndim == 6) {
          std::cout << "contract_tensor : " << std::endl;
          for (int i = 0; i < contract_tensor.rank(); ++i) std::cout << contract_tensor.extent(i) << ",";
          std::cout << std::endl;
          auto temp = pseudoInverse(n, rank, fast);
          std::cout << "pseudoInverse : " << std::endl;
          for (int i = 0; i < temp.rank(); ++i) std::cout << temp.extent(i) << ",";
          std::cout << std::endl;
          std::cout << "an : " << std::endl;
          for (int i = 0; i < an.rank(); ++i) std::cout << an.extent(i) << ",";
          std::cout << std::endl;
        }
        gemm(CblasNoTrans, CblasNoTrans, 1.0, contract_tensor, pseudoInverse(n, rank, fast), 0.0, an);
      }
#else
      matlab = false;
      if( !matlab){
        gemm(CblasNoTrans, CblasNoTrans, 1.0, contract_tensor, pseudoInverse(n, rank), 0.0, an);
      }
#endif
      //t2 = std::chrono::high_resolution_clock::now();
      //time = t2 - t1;
      //gemm_wPI += time.count();


      // Normalize the columns of the new factor matrix and update
      normCol(an);
      A[n] = an;
    }

  };

  template <typename Tensor, class ConvClass = NormCheck<Tensor>>
  class CP_THC_ALS : public CP<Tensor, ConvClass> {
  public:

    using CP<Tensor,ConvClass>::A;
    using CP<Tensor,ConvClass>::ndim;
    using CP<Tensor,ConvClass>::pseudoInverse;
    using CP<Tensor,ConvClass>::normCol;
    using CP<Tensor,ConvClass>::generate_KRP;
    using CP<Tensor,ConvClass>::generate_V;
    using CP<Tensor,ConvClass>::norm;

    CP_THC_ALS(Tensor & left, Tensor& right) : CP<Tensor, ConvClass>(left.rank() + right.rank()),
            tensor_ref_left(left),
            tensor_ref_right(right), ndimL(left.rank()){
      for(int i = 0; i < ndim; ++i){
        symm_dims.push_back(i);
      }
    }

    CP_THC_ALS(Tensor & left, Tensor& right, std::vector<int> symm) : CP<Tensor, ConvClass>(left.rank() + right.rank()),
            tensor_ref_left(left), tensor_ref_right(right), ndimL(left.rank()), symm_dims(symm){
      if(symm_dims.size() < ndim){
        BTAS_EXCEPTION("Symmetry must be defined for all dimensions");
      }
    }

#ifdef _HAS_INTEL_MKL
    double compute_PALS(std::vector<ConvClass> & converge_list, double RankStep = 0.5, int panels = 4, bool symm = false,
                        int max_als = 20,bool fast_pI = true, bool calculate_epsilon = false, bool direct = true) override{
      return 0;
    }
#endif //_HAS_INTEL_MKL


  protected:
    Tensor& tensor_ref_left;
    Tensor& tensor_ref_right;
    int ndimL;
    std::vector<int> symm_dims;
    Tensor non_tensor;  // Stores the L or R Tref contracted with factors and inverse

    void build(int rank, ConvClass &converge_test, bool direct, int max_als, bool calculate_epsilon, int step, double &epsilon,
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
        // start with left then do right
        auto ndimR = tensor_ref_right.rank();
        for(int i = 0; i < ndimL; i++){
          if(tensor_ref_left.extent(i) < SVD_rank){
            modes_w_dim_LT_svd.push_back(i);
          }
        }
        for(int i = 0; i < ndimR; i++){
          if(tensor_ref_right.extent(i) < SVD_rank){
            modes_w_dim_LT_svd.push_back(i + ndimL);
          }
        }

        for(int tensor = 0; tensor < 2; ++tensor){
          auto & tensor_ref = tensor == 0 ? tensor_ref_left : tensor_ref_right;
          auto ndim_curr = tensor_ref.rank();
          // Fill all factor matrices with their singular vectors,
          // because we contract X X^T (where X is reference tensor) to make finding
          // singular vectors an eigenvalue problem some factor matrices will not be
          // full rank;
          for(int i = 0; i < ndim_curr; i++){
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

            A[i + tensor * tensor_ref_left.rank()] = lambda;
          }
        }

        //srand(3);
        std::mt19937 generator(3);
        // Fill the remaining columns in the set of factor matrices with dimension < SVD_rank with random numbers
        for(auto& i: modes_w_dim_LT_svd){
          auto dim = i < ndimL ? i : i - ndimL;
          auto & tensor_ref = i < ndimL ? tensor_ref_left : tensor_ref_right;
          int R = tensor_ref.extent(dim);
          auto lower_bound = {0, R};
          auto upper_bound = {R, SVD_rank};
          auto view = make_view(A[i].range().slice(lower_bound, upper_bound), A[i].storage());
          std::normal_distribution<double> distribution(0, 2.0);
          for(auto iter = view.begin(); iter != view.end(); ++iter){
            *(iter) = distribution(generator);
          }
        }

        // Normalize the columns of the factor matrices and
        // set the values al lambda, the weigt of each order 1 tensor
        Tensor lambda(Range{Range1{SVD_rank}});
        A.push_back(lambda);
        for(auto i = 0; i < ndim; ++i){
          this->normCol(A[i]);
        }

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
          auto & tensor_ref = j < tensor_ref_left.rank() ? tensor_ref_left : tensor_ref_right;
          if (i == 0) {
            Tensor a(Range{tensor_ref.range(j), Range1{i + 1}});
            a.fill(rand());
            A.push_back(a);
            this->normCol(j);
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
              std::mt19937 generator(3);
              std::normal_distribution<double> distribution(0, 2);
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
        // compute the ALS of factor matrices with rank = i + 1.
        ALS(i + 1, converge_test, direct, max_als, calculate_epsilon, epsilon, fast_pI, symm);
      }
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
      non_tensor = Tensor(1,1);
      non_tensor.fill(0.0);
      // Until either the initial guess is converged or it runs out of iterations
      // update the factor matrices with or without Khatri-Rao product
      // intermediate
      bool is_converged = false;
      bool matlab = fast_pI;
      std::cout << "count\tfit\tFit Change" << std::endl;
      while(count < max_als && !is_converged){
        count++;
        this->num_ALS++;
        for (auto i = 0; i < ndim; i++) {
          if(symm_dims[i] == i) {
            direct(i, rank, symm, fast_pI, matlab, converge_test);
          } else {
            A[i] = A[symm_dims[i]];
          }
        }
        detail::get_fit(converge_test, epsilon);
        std::cout << count << "\t";
        is_converged = converge_test(A);
      }
    }

    // TODO Fix this
    void direct(int n, int rank, bool symm, bool & fast_pI, bool & matlab, ConvClass & converge_test) {

      // This forms B^X_r which needs to contract with B(n)^X_ab
      // Also need to check
      Tensor auxDim_vec(rank);
      auxDim_vec.fill(0.0);
      if(n == 0 || n == ndimL){
        dot_non_tensors(n == ndimL);
      }
      else{
        for(int i = 0; i < A[0].extent(0); ++i){
          for(int j = 0; j < rank; ++j){
            auxDim_vec(j) += A[0](i,j) * A[ndimL](i,j);
          }
        }
      }

      // This says if the mode of interest is on the left side
      bool left = n < ndimL ;

      auto & tensor_ref = left ? tensor_ref_left : tensor_ref_right;
      bool last_dim = n == (left ? ndimL - 1: ndim - 1);
      std::vector<int> dimensions;

      int aux_dim = tensor_ref.extent(0);
      int RHsize = tensor_ref.size() / aux_dim;
      auto R = tensor_ref.range();

      tensor_ref.resize(Range{Range1{aux_dim}, Range1{RHsize}});
      Tensor contract_tensor(RHsize, rank);

      // Contract B(n)^X_ab with other reference tensor B^X_r
      gemm(CblasTrans, CblasNoTrans, 1.0, tensor_ref, non_tensor, 0.0, contract_tensor);

      // resize the reference tensor
      tensor_ref.resize(R);

      int ndim_curr = tensor_ref.rank();
      int dim_in_A = left ? ndimL - 1 : ndim - 1;
      int contract_dim = ndim_curr - 1;

      int contract_size;
      int LHsize = contract_tensor.size() / rank;
      int offset_dim = A[n].extent(0);
      int pseudo_rank = rank;
      Tensor an(A[n].range());

      while(contract_dim >= 0){
        contract_size = A[dim_in_A].extent(0);
        LHsize /= contract_size;

        contract_tensor.resize(Range{Range1{LHsize}, Range1{contract_size}, Range1{pseudo_rank}});
        Tensor temp(LHsize, pseudo_rank);
        temp.fill(0.0);

        if(n == dim_in_A){
          pseudo_rank *= offset_dim;
          dimensions.pop_back();
        }

        else if(dim_in_A > n){
          for(int i = 0; i < LHsize; ++i){
            for(int j = 0; j < contract_size; ++j){
              for(int r = 0; r < pseudo_rank; ++r){
                temp(i,r) += contract_tensor(i,j,r) * A[dim_in_A](j,r);
              }
            }
          }
          contract_tensor = temp;
        }
        else{
          for(int i = 0; i < LHsize; ++i){
            for(int j = 0; j < contract_size; ++j){
              for(int k = 0; k < offset_dim; ++ k){
                for(int r = 0; r < rank; ++r){
                  temp(i, k * rank + r) += contract_tensor(i,j,k * rank + r) * A[dim_in_A](j,r);
                }
              }
            }
          }
          contract_tensor = temp;
        }
        --contract_dim;
        --dim_in_A;
      }
      if(n == 0 || n == ndim - 1){
        contract_tensor.resize(Range{Range1{rank}});
        auxDim_vec = contract_tensor;
        contract_tensor = Tensor(offset_dim, rank);
        auto & a = A[left ? 0 : ndim - 1];
        for(int i = 0; i < offset_dim; ++i){
          for(int r = 0; r < rank; ++r){
            contract_tensor(i,r) = auxDim_vec(r) *a(i,r);
          }
        }
      }
      else{
        contract_tensor.resize(Range{Range1{offset_dim}, Range1{rank}});
        for(int i = 0; i < offset_dim; ++i){
          for(int r = 0; r < rank; ++r){
            contract_tensor(i, r) *= auxDim_vec(r);
          }
        }
      }

      if(n + 1 == ndim) {
        detail::set_MtKRP(converge_test, contract_tensor);
      }

      gemm(CblasNoTrans, CblasNoTrans, 1.0, contract_tensor, this->pseudoInverse(n, rank, fast_pI), 0.0, an);

      this->normCol(an);
      A[n] = an;
    }


    void dot_non_tensors(bool left){
      auto & tensor_ref = left ? tensor_ref_left : tensor_ref_right;
      int dim_in_A = left ? ndimL - 1 : ndim - 1;
      int contract_dim = left ? ndimL - 1 : tensor_ref_right.rank() - 1;
      int contract_size = tensor_ref.extent(contract_dim);
      int LHsize = tensor_ref.size() / contract_size;
      int rank = A[0].extent(1);

      auto R = tensor_ref.range();
      non_tensor = Tensor(LHsize, rank);
      tensor_ref.resize(Range{Range1{LHsize}, Range1{contract_size}});
      gemm(CblasNoTrans, CblasNoTrans, 1.0, tensor_ref, A[dim_in_A], 0.0, non_tensor);
      tensor_ref.resize(R);
      contract_dim--; dim_in_A--;
      // This is easy never need to skip a dimension just hadamard contract all
      // the dimensions but the last one.
      for(;contract_dim > 0; contract_dim--, dim_in_A--){
        contract_size = A[dim_in_A].extent(0);
        LHsize /= contract_size;

        Tensor temp(LHsize, rank);
        non_tensor.resize(Range{Range1{LHsize}, Range1{contract_size}, Range1{rank}});
        for(int i = 0; i < LHsize; ++i){
          for(int j = 0; j < contract_size; ++j){
            for(int r = 0; r < rank; ++r){
              temp(i,r) += non_tensor(i,j,r) * A[dim_in_A](j,r);
            }
          }
        }
        non_tensor = temp;
      }
      return;
    }
  };
};// namespace btas

#endif //BTAS_GENERIC_CP_H
