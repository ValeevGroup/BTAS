//
// Created by Karl Pierce on 2/25/19.
//


#ifndef BTAS_GENERIC_CP_H
#define BTAS_GENERIC_CP_H

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

    template <typename Tensor>
    void set_MtKRPL(CoupledFitCheck<Tensor> & t, Tensor & tensor){
      t.set_MtKRPL(tensor);
    }

    template <typename Tensor>
    void set_MtKRPR(CoupledFitCheck<Tensor> & t, Tensor & tensor){
      t.set_MtKRPR(tensor);
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
    double compute_rank_random(int rank, ConvClass &converge_test, int step = 1, bool symm = false, int max_als = 1e4,
                               bool fast_pI = true, bool calculate_epsilon = false, bool direct = true) {
      if (rank <= 0) BTAS_EXCEPTION("Decomposition rank must be greater than 0");
      double epsilon = -1.0;
      build_random(rank, converge_test, direct, max_als, calculate_epsilon, step, epsilon,
            fast_pI, symm);
      std::cout << "Number of ALS iterations performed: " << num_ALS << std::endl;

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
    Tensor reconstruct() {
      if (A.empty()) BTAS_EXCEPTION(
              "Factor matrices have not been computed. You must first calculate CP decomposition.");
      std::vector<int> dims;
      for(auto i = 0; i < ndim; ++i){
        dims.push_back(i);
      }
      return btas::reconstruct(A, dims);
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

    virtual void build_random(int rank, ConvClass &converge_test, bool direct, int max_als, bool calculate_epsilon, double &epsilon,
                              bool & fast_pI, bool symm) = 0;

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
        if(*(A_ptr + i % rank) > 1e-12)
          *(Mat_ptr + i) /= *(A_ptr + i % rank);
        else
          *(Mat_ptr +i ) = 0;

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

};// namespace btas

#endif //BTAS_GENERIC_CP_H
