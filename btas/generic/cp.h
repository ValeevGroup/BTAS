//
// Created by Karl Pierce on 2/25/19.
//


#ifndef BTAS_GENERIC_CP_H
#define BTAS_GENERIC_CP_H

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
#include <btas/generic/linear_algebra.h>

namespace btas{
  namespace detail{

    // Functions that set the value of the original tensor
    // times the factor matrices (excluding one factor)
    // if the converge_class isn't a FitCheck do nothing
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

    // Functions that can get the fit \|X - \hat{X}\|_F where
    // \hat{X} is the CP approximation (epsilon), if
    // converge_class object isn't FitCheck do nothing
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

    template<typename Tensor>
    void get_fit(CoupledFitCheck<Tensor> & t, double & epsilon){
      epsilon = t.get_fit();
      return;
    }
  }//namespace detail

  /** \brief Base class to compute the Canonical Product (CP) decomposition of an order-N
    tensor.
    This is a virtual class and is constructed by its children to compute the CP decomposition
    using some type of solver.

    Synopsis:
    \code
    // Constructors
    CP A(ndim)                    // CP_ALS object with empty factor matrices

    // Operations
    A.compute_rank(rank, converge_test)             // Calls virtual build function
                                                    // to decompose to a specific rank
                                                    // has HOSVD option

    A.compute_rank_random(rank, converge_test)      // Calls virtual build_random function
                                                    // to decompose at specific rank,
                                                    // no HOSVD option

    A.compute_error(converge_test, omega)           // Calls the virtual build function to
                                                    // compute the CP decomposition to a 2-norm
                                                    // error < omega.

    A.compute_geometric(rank, converge_test, step)  // Calls the virtual build function to
                                                    // compute CP decomposition with rank that
                                                    // grows in geometric steps

    A.paneled_tucker_build(converge_test)           // computes CP_ALS of tensor to
                                                    // rank = 2 * max_dim(tensor)
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
  template <typename Tensor, class ConvClass>
  class CP {
  public:
    /// Create a generic CP object that stores the factor matrices,
    /// the number of iterations and the number of dimensions of the original
    /// tensor
    /// \param[in] ndim number of modes in the reference tensor.
    CP(int dims) : num_ALS(0) {
#if not defined(BTAS_HAS_LAPACKE)
      BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__, "CP decompositions requires LAPACKE");
#endif
      ndim = dims;
    }

    ~CP() = default;

    /// Computes decomposition of the order-N tensor \c tensor
    /// with CP rank = \c rank .
    /// Initial guess for factor matrices start at rank = ( 1 or \c SVD_rank)
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
    /// false && ConvClass != FitCheck.

    double compute_rank(int rank, ConvClass &converge_test, int step = 1,
                        bool SVD_initial_guess = false, int SVD_rank = 0, int max_als = 1e4,
                        bool fast_pI = false, bool calculate_epsilon = false, bool direct = true) {
      if (rank <= 0) BTAS_EXCEPTION("Decomposition rank must be greater than 0");
      if (SVD_initial_guess && SVD_rank > rank) BTAS_EXCEPTION("Initial guess is larger than the desired CP rank");
      double epsilon = -1.0;
      build(rank, converge_test, direct, max_als, calculate_epsilon, step, epsilon, SVD_initial_guess, SVD_rank,
            fast_pI);
      //std::cout << "Number of ALS iterations performed: " << num_ALS << std::endl;

      detail::get_fit(converge_test, epsilon);

      return epsilon;
    }

    /// Computes decomposition of the order-N tensor \c tensor
    /// with CP rank = \c rank factors initialized to rank \c rank
    /// using random numbers.

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged
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
    /// false && ConvClass != FitCheck.
    double compute_rank_random(int rank, ConvClass &converge_test, int max_als = 1e4,
                               bool fast_pI = false, bool calculate_epsilon = false, bool direct = true) {
      if (rank <= 0) BTAS_EXCEPTION("Decomposition rank must be greater than 0");
      double epsilon = -1.0;
      build_random(rank, converge_test, direct, max_als, calculate_epsilon, epsilon,
                   fast_pI);
      //std::cout << "Number of ALS iterations performed: " << num_ALS << std::endl;

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
    /// \param[in] max_als Max number of iterations allowed to converge the ALS
    /// approximation default = 1e4
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    /// default = true
    /// \param[in] direct Should the
    /// CP decomposition be computed without calculating the
    /// Khatri-Rao product? Default = true.
    /// \returns 2-norm
    /// error between exact and approximate tensor, -1 if calculate_epsilon =
    /// false && ConvClass != FitCheck.
    double compute_error(ConvClass &converge_test, double tcutCP = 1e-2, int step = 1,
                         int max_rank = 1e5, bool SVD_initial_guess = false, int SVD_rank = 0,
                         double max_als = 1e4, bool fast_pI = false, bool direct = true) {
      int rank = (A.empty()) ? ((SVD_initial_guess) ? SVD_rank : 1) : A[0].extent(0);
      double epsilon = tcutCP + 1;
      while (epsilon > tcutCP && rank < max_rank) {
        build(rank, converge_test, direct, max_als, true, step, epsilon, SVD_initial_guess, SVD_rank, fast_pI);
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
    /// \returns 2-norm
    /// error between exact and approximate tensor, -1 if calculate_epsilon =
    /// false && ConvClass != FitCheck.
    double compute_geometric(int desired_rank, ConvClass &converge_test, int geometric_step = 2,
                             bool SVD_initial_guess = false, int SVD_rank = 0, int max_als = 1e4,
                             bool fast_pI = false, bool calculate_epsilon = false, bool direct = true) {
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
              SVD_rank, fast_pI);
        if (geometric_step <= 1)
          rank++;
        else
          rank *= geometric_step;
      }

      detail::get_fit(converge_test, epsilon);
      return epsilon;
    }

    /// virtual function implemented in solver
    /// Computes decomposition of the order-N tensor \c tensor
    /// with rank = \c RankStep * \c panels *  max_dim(reference_tensor) + max_dim(reference_tensor)
    /// Initial guess for factor matrices start at rank = max_dim(reference_tensor)
    /// and builds rank \c panel times by \c RankStep * max_dim(reference_tensor) increments

    /// \param[in] converge_test Test to see if ALS is converged
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
    /// \returns 2-norm
    /// error between exact and approximate tensor, -1 if calculate_epsilon =
    /// false && ConvClass != FitCheck.
    virtual double compute_PALS(std::vector<ConvClass> & converge_list, double RankStep = 0.5, int panels = 4,
                         int max_als = 20,bool fast_pI = false, bool calculate_epsilon = false, bool direct = true) = 0;

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

    /// Default function, uses the factor matrices from the CP
    /// decomposition and reconstructs the
    /// approximated tensor.
    /// Assumes that $T_{(A)} = A (B \odot C \odot D \dots) ^T$

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

    // For debug purposes
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
    int num_ALS;                      // Number of ALS iterations
    std::vector<Tensor> A;            // Factor matrices
    int ndim;                         // Modes in the reference tensor
    std::vector<int> symmetries;      // Symmetries of the reference tensor

    /// Virtual function. Solver classes should implement a build function to
    /// generate factor matrices then compute the CP decomposition
    /// This function should have options for HOSVD and for building rank by \c step increments

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged.
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
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    virtual void build(int rank, ConvClass &converge_test, bool direct, int max_als, bool calculate_epsilon, int step, double &epsilon,
                  bool SVD_initial_guess, int SVD_rank, bool & fast_pI) = 0;
    
    /// Virtual function. Solver classes should implement a build function to generate factor matrices then compute the CP decomposition
    /// Create a rank \c rank initial guess using
    /// random numbers from a uniform distribution

    /// \param[in] rank The rank of the CP decomposition.
    /// \param[in] converge_test Test to see if ALS is converged.
    /// \param[in] direct The CP decomposition be computed without calculating the
    /// Khatri-Rao product?
    /// \param[in] max_als If CP decomposition is to finite
    /// error, max_als is the highest rank approximation computed before giving up
    /// on CP-ALS.
    /// \param[in] calculate_epsilon Should the 2-norm
    /// error be calculated \f$ ||T_{\rm exact} - T_{\rm approx}|| = \epsilon \f$ .
    /// \param[in, out] epsilon The 2-norm
    /// error between the exact and approximated reference tensor
    /// \param[in] SVD_initial_guess build inital guess from left singular vectors
    /// \param[in] SVD_rank rank of the initial guess using left singular vector
    /// \param[in] fast_pI Should the pseudo inverse be computed using a fast cholesky decomposition
    virtual void build_random(int rank, ConvClass &converge_test, bool direct, int max_als, bool calculate_epsilon, double &epsilon,
                              bool & fast_pI) = 0;

    /// Generates V by first Multiply A^T.A then Hadamard product V(i,j) *=
    /// A^T.A(i,j);
    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    /// \param[in] lambda regularization parameter, lambda is added to the diagonal of V
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
        *(V_ptr + j * rank + j) += lambda;
      }

      return V;
    }

    /// Keep track of the Left hand Khatri-Rao product of matrices and
    /// Continues to multiply be right hand products, skipping
    /// the matrix at index n.
    /// \param[in] n The mode being optimized, all other modes held constant
    /// \param[in] rank The current rank, column dimension of the factor matrices
    /// \param[in] forward Should the Khatri-Rao product move through the factor
    /// matrices in the forward (0 to ndim) or backward (ndim to 0) direction
    /// \return the Khatri-Rao product of the factor matrices excluding the nth factor
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

    /// \param[in] factor Which factor matrix to normalize, returns
    /// the \c factor factor matrix with all columns normalized.
    /// \return The column norms of the \c factor factor matrix

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

    /// Calculates the column norms of a matrix and saves the norm values into
    /// lambda tensor (last matrix in the A)

    /// \param[in, out] Mat The matrix whose column will be normalized, return
    /// \c Mat with all columns normalized

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

    /// Trying to solve Ax = B
    /// First try Cholesky to solve this problem directly
    /// second tryfast pseudo-inverse algorithm described in
    /// https://arxiv.org/pdf/0804.4809.pdf
    /// If all else fails use SVD

    /// \param[in] mode_of_A The mode being optimized used to compute hadamard LHS (V) of ALS problem (Vx = B)
    /// \param[in,out] fast_pI If true, try to compute the pseudo inverse via fast LU decomposition, else use SVD;
    ///                on return reports whether the fast route was used.
    /// \param[in, out] cholesky If true, try to solve the linear equation Vx = B (the ALS problem)
    ///                using a Cholesky decomposition (lapacke subroutine) on return reports if
    ///                inversion was successful.
    /// \param[in, out] B In: The RHS of the ALS problem ( Vx = B ). Out: The solved linear equation
    ///                     \f$ V^{-1} B \f$
    /// \param[in] lambda Regularization parameter lambda is added to the diagonal of V
    void psuedoinverse_helper(int mode_of_A, bool & fast_pI,
                                bool & cholesky, Tensor & B,
                                double lambda = 0.0){
      if(B.empty()){
        BTAS_EXCEPTION("pseudoinverse helper solves Ax = B.  B cannot be an empty tensor");
      }

      auto rank = A[0].extent(1);
      auto a = this->generate_V(mode_of_A, rank, lambda);

      if(cholesky) {
        cholesky = cholesky_inverse(a, B);
        return;
      }
      auto pInv = pseudoInverse(a, fast_pI);
      Tensor an(B.extent(0), rank);
      gemm(CblasNoTrans, CblasNoTrans, 1.0, B, pInv, 0.0, an);
      B = an;
    }
  };
};// namespace btas

#endif //BTAS_GENERIC_CP_H
