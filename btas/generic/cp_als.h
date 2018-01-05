#ifndef BTAS_GENERIC_CP_ALS_H
#define BTAS_GENERIC_CP_ALS_H

/*!Canonical Product*/

#include <algorithm>
#include <iostream>
#include <stdlib.h>

#include <btas/btas.h>
#include <btas/error.h>
#include "khatri_rao_product.h"
#include "core_contract.h"
#include "flatten.h"
#include "randomized.h"
#include "swap.h"
#include "tucker.h"

namespace btas {

template <typename Tensor> class cp_als {
public:
  cp_als(Tensor &tensor)
      : tensor_ref(tensor), ndim(tensor_ref.rank()), size(tensor_ref.size()) {

#if not defined(BTAS_HAS_CBLAS) || not defined(_HAS_INTEL_MKL)
    BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__,
                           "CP_ALS requires LAPACKE or mkl_lapack");
#endif
#ifdef _HAS_INTEL_MKL
#include <mkl_trans.h>
#endif
  }

  ~cp_als() = default;

  /// Computes decomposition of Nth order tensor T 
  /// with CP rank = rank\n
  /// Initial guess for factor matrices start at rank = 1 
  /// and build to rank = rank by increments of step, to minimize 
  /// error.
  double compute(const int rank, const bool direct = true,
                 const bool calculate_epsilon = false, const double max_rank = 1e5,
                 const int step = 1, const double tcutALS = 0.1) {
    double epsilon = 0.0;
    build(rank, direct, max_rank, calculate_epsilon, step, tcutALS, epsilon);
    return epsilon;
  }

  /// Computes the decomposition of Nth order tensor T 
  /// to rank <= max_rank with \n
  /// \t |T_exact - T_approx|_F <= tcutCP \n
  /// with rank increasing each iteration by step.
  double compute(const double tcutCP = 1e-2, const bool direct = true,
                 const double max_rank = 1e5, const int step = 1,
                 const double tcutALS = 0.1) {
    int rank = (A.empty()) ? 0 : A[0].extent(0);
    double epsilon = tcutCP + 1;
    while (epsilon > tcutCP && rank < max_rank) {
      rank += step;
      build(rank, direct, max_rank, true, step, tcutALS, epsilon);
    }
    return epsilon;
  }
  /// Computes decomposition of Nth order tensor T 
  /// with CP rank <= desired_rank\n
  /// Initial guess for factor matrices start at rank = 1 
  /// and build to rank = rank by geometric steps of geometric_step, to minimize 
  /// error.
  double compute_geometric(const int desired_rank, int geometric_step = 2, 
                           const bool direct = false, const bool calculate_epsilon = false,
                           const double max_rank = 1e5, const double tcutALS = 0.1){
    if(geometric_step <= 0){
      std::cout << "The step size must be larger than 0" << std::endl;
      return 0;
    }
    double epsilon = 0.0;
    int rank = 1;
    while (rank <= desired_rank && rank < max_rank){
      build(rank, direct, max_rank, calculate_epsilon, geometric_step, tcutALS, epsilon);
      if(geometric_step == 1)
        rank++;
      else
        rank *= geometric_step;
    }
    return epsilon;
  }
#ifdef _HAS_INTEL_MKL
  /// Requires MKL. Computes an approximate core tensor using 
  /// Tucker decomposition, i.e.  \n
  ///  T(I_1, I_2, I_3) --> T(R1, R2, R3) \n
  /// Where R1 < I_1, R2 < I_2 and R3 < I_3
  /// see <a href="http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7516088"> \n
  /// Using this approximation the CP decomposition is computed to 
  /// either finite error or finite rank. \n
  /// Default settings calculate to finite error.
  /// factor matrices are scaled by the Tucker transformations.
  double compress_compute_tucker(double tcutSVD, const bool opt_rank = true,
                               const double tcutCP = 1e-2, const int rank = 0,
                               const bool direct = true,
                               const bool calculate_epsilon = false,
                               const double max_rank = 1e5, const int step = 1,
                               const double tcutALS = .01) {
    // Tensor compression
    std::vector<Tensor> transforms;
    tucker_compression(tensor_ref, tcutSVD, transforms);
    size = tensor_ref.size();
    double epsilon = -1.0;

    // CP decomposition
    if (opt_rank)
      epsilon = compute(tcutCP, direct, max_rank, step, tcutALS);
    else if (rank == 0) {
      std::cout << "Must specify a rank > 0" << std::endl;
      return epsilon;
    } else
      epsilon = compute(rank, direct, calculate_epsilon, max_rank, step, tcutALS);

    //scale factor matrices
    for(int i = 0; i < ndim; i++){
      Tensor tt(transforms[i].extent(0), A[i].extent(1));
      gemm(CblasNoTrans, CblasNoTrans, 1.0, transforms[i], A[i], 0.0, tt);
      A[i] = tt;
    }

    return epsilon;
  }
  /// Requires MKL. Computes an approximate core tensor using 
  /// random projection, i.e.  \n
  ///  T(I_1, I_2, I_3) --> T(R, R, R) \n
  /// Where R < I_1, R < I_2 and R < I_3
  /// see <a href="https://arxiv.org/pdf/1703.09074.pdf"> \n
  /// Using this approximation the CP decomposition is computed to 
  /// either finite error or finite rank. \n
  /// Default settings calculate to finite error.\n
  /// Factor matrices are scaled by randomized transformation.
  double compress_compute_rand(int desired_compression_rank,
                             const int oversampl = 10, const int powerit = 2,
                             const bool opt_rank = true,
                             const double tcutCP = 1e-2, const int rank = 0,
                             const bool direct = true,
                             const bool calculate_epsilon = false,
                             const double max_rank = 1e5, const int step = 1,
                             const double tcutALS = .1) {
    std::vector<Tensor> transforms;
    randomized_decomposition(tensor_ref, desired_compression_rank, transforms,
                             oversampl, powerit);
    size = tensor_ref.size();
    double epsilon = -1.0;

    if (opt_rank)
      epsilon = compute(tcutCP, direct, max_rank, step, tcutALS);
    else if (rank == 0) {
      std::cout << "Must specify a rank > 0" << std::endl;
      return epsilon;
    } else
      epsilon = compute(rank, direct, calculate_epsilon, max_rank, step, tcutALS);

    //scale factor matrices
    for(int i = 0; i < ndim; i++){
      Tensor tt(transforms[i].extent(0), A[i].extent(1));
      gemm(CblasNoTrans, CblasNoTrans, 1.0, transforms[i], A[i], 0.0, tt);
      A[i] = tt;
    }

    return epsilon;
  }
#endif //_HAS_INTEL_MKL

  //returns 
  std::vector<Tensor> get_factor_matrices(){
    if(A != NULL)
      return A;
    else
      BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__,
                           "Attempting to use a NULL object first compute CP decomposition");
  }
private:
  std::vector<Tensor> A; // vector of factor matrices
  Tensor &tensor_ref; // this is a reference.
  const int ndim;
  int size;

  // creates factor matricies starting with R=1 and moves to R = rank
  // Where R is the column dimension of the factor matrices.
  void build(const int rank, const bool dir, const int max_rank, const bool calculate_epsilon,
             const int step, const double tcutALS, double &epsilon) {
    for (auto i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += step) {
      for (auto j = 0; j < ndim; ++j) { // select a factor matrix
        if (i == 0) { // creates a factor matrix when A is empty
          Tensor a(Range{tensor_ref.range(j), Range1{i + 1}});
          a.fill(rand());
          normCol(a, i);
          A.push_back(a);
          if (j + 1 == ndim) {
            Tensor lam(Range{Range1{i + 1}});
            A.push_back(lam);
          }
        } else { // builds onto factor matrices when R > 1
                 // This could be done by A -> A^T then adding a row then (A')^T
                 // -> A'
          Tensor b(Range{A[0].range(0), Range1{i + 1}});
          b.fill(rand());
          for (int l = A[0].extent(1); l < i + 1; ++l)
            normCol(b, l);
          for (int k = 0; k < b.extent(0); k++)
            for (int l = 0; l < A[0].extent(1); l++)
              b(k, l) = A[0](k, l);

          A.erase(A.begin());
          A.push_back(b);
          if (j + 1 == ndim) {
            b.resize(Range{Range1{i + 1}});
            for (int k = 0; k < A[0].extent(0); k++)
              b(k) = A[0](k);
            A.erase(A.begin());
            A.push_back(b);
          }
        }
      }
      ALS(i + 1, dir, max_rank, calculate_epsilon, tcutALS,
          epsilon); // performs the least squares minimization to generate the
                      // best CP rank i+1 approximation
    }
  }

  // performs the ALS method to minimize the loss function for a single rank
  void ALS(const int rank, const bool dir, const int max_rank, const bool calculate_epsilon,
           const double tcutALS, double &epsilon) {
    auto count = 0;
    double test = 1.0;
    while (count <= max_rank && test > tcutALS) {
      count++;
      test = 0.0;
      for (auto i = 0; i < ndim; i++) {
        if (dir)
          direct(i, rank, test);
        else
          update_w_KRP(i, rank, test); // generates the next iteration of factor
                                       // matrix to minimize the least squares
                                       // problem.
      }
    }
    // only checks loss function if required
    if (calculate_epsilon) {
      Tensor oldmat(tensor_ref.extent(0), size / tensor_ref.extent(0));
      for (int i = 0; i < rank; i++) {
        scal(A[0].extent(0), A[ndim](i), std::begin(A[0]) + i, rank);
      }
      gemm(CblasNoTrans, CblasTrans, 1.0, A[0], generate_KRP(0, rank, false),
           0.0, oldmat);
      for (int i = 0; i < rank; i++) {
        scal(A[0].extent(0), 1 / A[ndim](i), std::begin(A[0]) + i, rank);
      }
      epsilon = (norm(flatten(tensor_ref, 0) - oldmat));
    }
  }

  // This update requires computation of the khatri-rao product every time its
  // called
  void update_w_KRP(int n, int rank, double &test) {
    // multiply the components together to get the least squares minimization
    Tensor temp(A[n].extent(0), rank);
    Tensor an(A[n].range());
#ifdef _HAS_INTEL_MKL
    auto KhatriRao = generate_KRP(n, rank, true);
    swap_to_first(tensor_ref, n);
    std::vector<size_t> tref_dims, KRP_dims, An_dims;

    // resize KRP
    for (int i = 1; i < ndim; i++) {
      KRP_dims.push_back(tensor_ref.extent(i));
    }
    KRP_dims.push_back(rank);
    KhatriRao.resize(KRP_dims);
    KRP_dims.clear();

    // build contraction vectors
    An_dims.push_back(0);
    An_dims.push_back(ndim);
    tref_dims.push_back(0);
    for (int i = 1; i < ndim; i++) {
      tref_dims.push_back(i);
      KRP_dims.push_back(i);
    }
    KRP_dims.push_back(ndim);

    // contract
    contract(1.0, tensor_ref, tref_dims, KhatriRao, KRP_dims, 0.0, temp,
             An_dims);
    swap_to_first(tensor_ref, n, true);

#else // BTAS_HAS_CBLAS
    gemm(CblasNoTrans, CblasNoTrans, 1.0, flatten(tensor_ref, n),
         generate_KRP(n, rank, true), 0.0, temp);
#endif
    gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, pseudoInverse(n, rank), 0.0,
         an);
    for (auto l = 0; l < rank; ++l)
      A[ndim](l) = normCol(an, l);
    test += norm(A[n] - an);
    A[n] = an;
  }
  // No Khatri-Rao product computed, immediate contraction
  // N = 4, n = I2
  // T(I0, I1, I2, I3) --> T(I0 I1 I2, I3)
  // T(I0 I1 I2, I3) x A( I3, R) ---> T(I0 I1 I2, R)
  // T(I0 I1 I2, R) --> T(I0 I1, I2, R) --> T(I0 I1, I2 R)
  // T(I0 I1, I2 R) --> T(I0, I1, I2 R) (x) A(I1, R) --> T(I0, I2 R)
  // T(I0, I2, R) (x) T(I0, R) --> T(I2, R)

  // N = 3, n = I2
  // T(I0, I1, I2) --> T(I0, I1 I2)
  // (T(I0, I1 I2))^T A(I0, R) --> T(I1 I2, R)
  // T(I1, I2, R) (x) T(I1, R) --> T(I2, R)
  void direct(const int n, const int rank, double &test) {
    int LH_size = size;
    int contract_dim = ndim - 1;
    int pseudo_rank = rank;
    Tensor temp(1, 1);
    Range R = tensor_ref.range();
    Tensor an(A[n].range());
    if (n < ndim - 1) {
      tensor_ref.resize(Range{Range1{size / tensor_ref.extent(contract_dim)},
                              Range1{tensor_ref.extent(contract_dim)}});
      temp.resize(Range{Range1{tensor_ref.extent(0)}, Range1{rank}});
      gemm(CblasNoTrans, CblasNoTrans, 1.0, tensor_ref, A[contract_dim], 0.0,
           temp);
      tensor_ref.resize(R);
      LH_size /= tensor_ref.extent(contract_dim);
      contract_dim--;
      while (contract_dim > 0) {
        temp.resize(Range{Range1{LH_size / tensor_ref.extent(contract_dim)},
                          Range1{tensor_ref.extent(contract_dim)},
                          Range1{pseudo_rank}});
        Tensor contract_tensor(
            Range{Range1{temp.extent(0)}, Range1{temp.extent(2)}});
        if (n == contract_dim) {
          pseudo_rank *= tensor_ref.extent(contract_dim);
        } else if (contract_dim > n) {
          for (int i = 0; i < temp.extent(0); i++)
            for (int r = 0; r < rank; r++)
              for (int j = 0; j < temp.extent(1); j++)
                contract_tensor(i, r) += temp(i, j, r) * A[contract_dim](j, r);
          temp = contract_tensor;
        } else {
          for (int i = 0; i < temp.extent(0); i++)
            for (int r = 0; r < rank; r++)
              for (int k = 0; k < tensor_ref.extent(n); k++)
                for (int j = 0; j < temp.extent(1); j++)
                  contract_tensor(i, k * rank + r) +=
                      temp(i, j, k * rank + r) * A[contract_dim](j, r);
          temp = contract_tensor;
        }

        LH_size /= tensor_ref.extent(contract_dim);
        contract_dim--;
      }
      if (n != 0) {
        temp.resize(Range{Range1{tensor_ref.extent(0)},
                          Range1{tensor_ref.extent(n)}, Range1{rank}});
        Tensor contract_tensor(Range{Range1{temp.extent(1)}, Range1{rank}});
        for (int i = 0; i < temp.extent(0); i++)
          for (int r = 0; r < rank; r++)
            for (int j = 0; j < temp.extent(1); j++)
              contract_tensor(j, r) += A[0](i, r) * temp(i, j, r);
        temp = contract_tensor;
      }
      gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, pseudoInverse(n, rank), 0.0,
           an);
    } else {
      contract_dim = 0;
      tensor_ref.resize(Range{Range1{tensor_ref.extent(0)},
                              Range1{size / tensor_ref.extent(0)}});
      temp.resize(Range{Range1{rank}, Range1{size / tensor_ref.extent(0)}});
      gemm(CblasTrans, CblasNoTrans, 1.0, A[0], tensor_ref, 0.0, temp);
      tensor_ref.resize(R);
      LH_size /= tensor_ref.extent(contract_dim);
      contract_dim++;
      while (contract_dim < ndim - 1) {
        temp.resize(Range{Range1{rank}, Range1{tensor_ref.extent(contract_dim)},
                          Range1{LH_size / tensor_ref.extent(contract_dim)}});
        Tensor contract_tensor(Range{Range1{rank}, Range1{temp.extent(2)}});
        for (int i = 0; i < temp.extent(2); i++)
          for (int r = 0; r < rank; r++)
            for (int j = 0; j < temp.extent(1); j++)
              contract_tensor(r, i) += temp(r, j, i) * A[contract_dim](j, r);
        temp = contract_tensor;
        LH_size /= tensor_ref.extent(contract_dim);
        contract_dim++;
      }
      gemm(CblasTrans, CblasNoTrans, 1.0, temp, pseudoInverse(n, rank), 0.0,
           an);
    }
    for (auto l = 0; l < rank; ++l)
      A[ndim](l) = normCol(an, l);
    test += norm(A[n] - an);
    A[n] = an;
  }

  Tensor generate_V(const int n, const int rank) {
    // To generate V first Multiply A^T.A then Hadamard product V(i,j) *=
    // A^T.A(i,j);
    Tensor V(rank, rank);
    V.fill(1.0);
    for (auto j = 0; j < ndim; ++j) {
      if (j != n) {
        Tensor T = A.at(j);
        Tensor lhs_prod(rank, rank);
        gemm(CblasTrans, CblasNoTrans, 1.0, T, T, 0.0, lhs_prod);
        for (int i = 0; i < rank; i++)
          for (int k = 0; k < rank; k++)
            V(i, k) *= lhs_prod(i, k);
      }
    }
    return V;
  }

  Tensor generate_KRP(const int n, const int rank, const bool forward) {
    // Keep track of the Left hand Khatri-Rao product of matrices and
    // Continues to multiply be right hand products, skipping
    // the matrix at index n.
    // The product works backwards from Last factor matrix to the first.
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
    } else {
      for (auto i = ndim - 1; i > -1; --i) {
        if ((i == ndim - 1 && n != ndim - 1) ||
            (i == ndim - 2 && n == ndim - 1)) {
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

  double normCol(const int factor, const int col) {
    const double *AF_ptr = A[factor].data() + col;
    double norm = sqrt(dot(A[factor].extent(0), AF_ptr, A[factor].extent(1),
                           AF_ptr, A[factor].extent(1)));
    scal(A[factor].extent(0), 1 / norm, std::begin(A[factor]) + col,
         A[factor].extent(1));
    return norm;
  }

  double normCol(Tensor &Mat, int col) {
    const double *Mat_ptr = Mat.data() + col;
    double norm = sqrt(
        dot(Mat.extent(0), Mat_ptr, Mat.extent(1), Mat_ptr, Mat.extent(1)));
    scal(Mat.extent(0), 1 / norm, std::begin(Mat) + col, Mat.extent(1));
    return norm;
  }

  double norm(const Tensor &Mat) { return sqrt(dot(Mat, Mat)); }

  // SVD referencing code from
  // http://www.netlib.org/lapack/explore-html/de/ddd/lapacke_8h_af31b3cb47f7cc3b9f6541303a2968c9f.html
  Tensor pseudoInverse(int n, const int R) { // works no error
    auto a = generate_V(n, R);
    Tensor s(Range{Range1{R}});
    Tensor U(Range{Range1{R}, Range1{R}});
    Tensor Vt(Range{Range1{R}, Range1{R}});
#ifdef _HAS_INTEL_MKL
    double worksize;
    double *work = &worksize;
    lapack_int lwork = -1;
    lapack_int info = 0;

    char len = 1;
    char A = 'A';

    // Call dgesvd with lwork = -1 to query optimal workspace size:

    info =
        LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, A, A, R, R, a.data(), R, s.data(),
                            U.data(), R, Vt.data(), R, &worksize, lwork);
    if (info)
      ;
    lwork = (lapack_int)worksize;
    work = (double *)malloc(sizeof(double) * lwork);

    info =
        LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, A, A, R, R, a.data(), R, s.data(),
                            U.data(), R, Vt.data(), R, work, lwork);
    if (info)
      ;

    free(work);
#else // BTAS_HAS_CBLAS
    gesvd('A', 'A', a, s, U, Vt);
#endif
    double lr_thresh = 1e-13;
    Tensor s_inv(Range{Range1{R}, Range1{R}});
    for (auto i = 0; i < R; ++i) {
      if (s(i) > lr_thresh)
        s_inv(i, i) = 1 / s(i);
      else
        s_inv(i, i) = s(i);
    }
    s.resize(Range{Range1{R}, Range1{R}});

    gemm(CblasNoTrans, CblasNoTrans, 1.0, U, s_inv, 0.0, s);
    gemm(CblasNoTrans, CblasNoTrans, 1.0, s, Vt, 0.0, U);

    return U;
  }
};

} // namespace btas
#endif // BTAS_GENERIC_CP_ALS_H
