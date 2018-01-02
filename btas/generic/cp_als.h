// std
#ifndef CP
#define CP

#include <algorithm>
#include <array>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <stdlib.h>
// BTAS
#include <btas/generic/Khatri_Rao_Product.h>
#include <btas/generic/flatten.h>
#include <btas/generic/swap.h>
#include <btas/btas.h>
#include <btas/error.h>
#ifdef _HAS_INTEL_MKL
#include <mkl_trans.h>
#endif

namespace btas {

template <typename Tensor> class CP_ALS {
public:
  using value_type = typename Tensor::value_type;
  typedef typename std::vector<value_type>::const_iterator iterator;

  CP_ALS(Tensor &tensor)
      : tensor_ref(tensor), ndim(tensor_ref.rank()), size(tensor_ref.size()) {

#if not defined(BTAS_HAS_CBLAS) || not defined(_HAS_INTEL_MKL)
    BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__,
                           "CP_ALS requires LAPACKE or mkl_lapack");
#endif
#ifdef _HAS_INTEL_MKL
#include <mkl_trans.h>
#endif
  }

  ~CP_ALS() = default;

  // Flat is only true when tensor is 4 dimensional
  double compute(const int rank, const bool direct = true,
                 const bool r_test = false, const double max_R = 1e5,
                 const int skip = 1, const double ep_ALS = 0.1) {
    // most basic compute
    // This will optimize the tensor approximation with rank r for a single R
    // and return the vector of factor matrices
    double rank_test = 0.0;
    Build(rank, direct, max_R, r_test, skip, ep_ALS, rank_test);
    return rank_test;
  }

  double compute(const double epsilon = 1e-2, const bool direct = true,
                 const double max_R = 1e5, const int skip = 1,
                 const double ep_ALS = 0.1) {
    // this method computes the rank decompositions from r=1 to max_R until
    // the Frobenius norm difference between the initial tensor and the
    // decomposition is below some threshold.
    int rank = (A.empty()) ? 0 : A[0].extent(0);
    double rank_test = epsilon + 1;
    while (rank_test > epsilon && rank < max_R) {
      rank += skip;
      Build(rank, direct, max_R, true, skip, ep_ALS, rank_test);
      std::cout << "With rank " << rank << " there is a difference of "
                << rank_test << std::endl;
    }
    std::cout << "The decomposition finished with rank " << rank
              << " and epsilon " << rank_test << std::endl;
    return rank_test;
  }
#ifdef _HAS_INTEL_MKL
  void compress_compute_tucker(double epsilon_svd, const bool opt_rank = true,
                               const double epsilon = 1e-2, const int rank = 0,
                               const bool direct = true,
                               const bool r_test = false,
                               const double max_R = 1e5, const int skip = 1,
                               const double ep_ALS = .01) {
    // Tensor compression
    std::vector<Tensor> transforms;
    Tensor test = tensor_ref;
    std::cout << tensor_ref << std::endl;
    Tucker_compression(epsilon_svd, transforms);
    Tensor help = Tensor(tensor_ref.range());
    help.fill(0.0);
    std::cout << tensor_ref << std::endl;
    // CP decomposition
    if (opt_rank)
      compute(epsilon, direct, max_R, skip, ep_ALS);
    else if (rank == 0) {
      std::cout << "Must specify a rank > 0" << std::endl;
      return;
    } else
      compute(rank, direct, r_test, max_R, skip, ep_ALS);

    for (int i = 0; i < A[0].extent(0); i++)
      for (int j = 0; j < A[1].extent(0); j++)
        for (int k = 0; k < A[0].extent(1); k++)
          if (tensor_ref.rank() == 2)
            help(i, j) += A[ndim](k) * A[0](i, k) * A[1](j, k);
          else
            for (int l = 0; l < A[2].extent(0); l++)
              help(i, j, l) +=
                  A[ndim](k) * A[0](i, k) * A[1](j, k) * A[2](l, k);
    tensor_ref = help;
    for (int i = 0; i < ndim; i++) {
      contract_core(transforms[i], i, false);
    }
    std::cout << "norm btwn tensor_ref and test " << norm(tensor_ref - test)
              << std::endl;
  }

  void compress_compute_rand(int desired_compression_rank,
                             const int oversampl = 10, const int powerit = 2,
                             const bool opt_rank = true,
                             const double epsilon = 1e-2, const int rank = 0,
                             const bool direct = true,
                             const bool r_test = false,
                             const double max_R = 1e5, const int skip = 1,
                             const double ep_ALS = .1) {
    Tensor test = tensor_ref;
    std::vector<Tensor> transforms;
    Randomized_Decomposition(desired_compression_rank, transforms, oversampl,
                             powerit);
    if (opt_rank)
      compute(epsilon, direct, max_R, skip, ep_ALS);
    else if (rank == 0) {
      std::cout << "Must specify a rank > 0" << std::endl;
      return;
    } else
      compute(rank, direct, r_test, max_R, skip, ep_ALS);
    std::vector<double> norms(ndim, 0);
    for (int i = 0; i < ndim; i++)
      norms[i] = norm(A[i]);
    for (int i = 0; i < ndim; i++) {
      Tensor hold(transforms[i].extent(0), A[i].extent(1));
      gemm(CblasNoTrans, CblasNoTrans, 1.0, transforms[i], A[i], 0.0, hold);
      A[i] = hold;
      for (int j = 0; j < A[i].extent(1); j++)
        A[ndim](j) = normCol(i, j);
    }
    tensor_ref = test;
    size = 1;
    for (int i = 0; i < ndim; i++)
      size *= tensor_ref.extent(i);
  }
#endif //_HAS_INTEL_MKL

  std::vector<Tensor> A; // vector of factor matrices

private:
  ///////Global variables//////////

  Tensor &tensor_ref; // this is a reference.
  const int ndim;
  int size;

  // This is the rank r residual between tensor_ref and A

  ///////CP-ALS Specific functions to build and optimize guesses /////////

  // creates factor matricies starting with R=1 and moves to R = rank
  // Where R is the column dimension of the factor matrices.
  void Build(const int rank, const bool dir, const int max_R, const bool r_test,
             const int skip, const double ep_ALS, double &rank_test) {
    std::cout.precision(8);
    for (auto i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += skip) {
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
      ALS(i + 1, dir, max_R, r_test, ep_ALS,
          rank_test); // performs the least squares minimization to generate the
                      // best CP rank i+1 approximation
    }
  }

  // performs the ALS method to minimize the loss function for a single rank
  void ALS(const int rank, const bool dir, const int max_R, const bool r_test,
           const double ep_ALS, double &rank_test) {
    auto count = 0;
    double test = 1.0;
    while (count <= max_R && test > ep_ALS) {
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
    if (r_test) {
      Tensor oldmat(tensor_ref.extent(0), size / tensor_ref.extent(0));
      for (int i = 0; i < rank; i++) {
        scal(A[0].extent(0), A[ndim](i), std::begin(A[0]) + i, rank);
      }
      gemm(CblasNoTrans, CblasTrans, 1.0, A[0], generate_KRP(0, rank, false),
           0.0, oldmat);
      for (int i = 0; i < rank; i++) {
        scal(A[0].extent(0), 1 / A[ndim](i), std::begin(A[0]) + i, rank);
      }
      rank_test = (norm(Flatten(tensor_ref, 0) - oldmat));
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
    gemm(CblasNoTrans, CblasNoTrans, 1.0, Flatten(tensor_ref, n),
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
        // vdMul(rank*rank, V.data(), lhs_prod.data(), V.data());
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
    if (forward) { // forward direction
      for (auto i = 0; i < ndim; ++i) {
        if ((i == 0 && n != 0) || (i == 1 && n == 0)) {
          left_side_product = A.at(i);
        } else if (i != n) {
          KhatriRaoProduct(left_side_product, A[i], temp);
          left_side_product = temp;
        }
      }
    } else { // backward direction
      for (auto i = ndim - 1; i > -1; --i) {
        if ((i == ndim - 1 && n != ndim - 1) ||
            (i == ndim - 2 && n == ndim - 1)) {
          left_side_product = A.at(i);
        }

        else if (i != n) {
          KhatriRaoProduct<Tensor>(left_side_product, A[i], temp);
          left_side_product = temp;
        }
      }
    }
    return left_side_product;
  }

  void print(Tensor &A, int cols) {
    auto n_cols = 0;
    for (auto &beta : tensor_ref) {
      std::cout << beta << ", ";
      n_cols++;
      if (n_cols == cols) {
        n_cols = 0;
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
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
    // R is the rank *a is the pointer to the left
    // most peice of data in the matrix list
    // This is not finished need to dot UsVT together to get A back.
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

// Modifies tensor_ref to reduce compress the data without
// Tucker3 decomposition using random distributions of the
// columns and a power sampling method
#ifdef _HAS_INTEL_MKL
  void Randomized_Decomposition(int des_rank, std::vector<Tensor> &transforms,
                                int oversampl = 10, int powerit = 2) {
    des_rank += oversampl;
    for (int i = 0; i < ndim; i++) {
      auto Bn = Flatten(tensor_ref, i);
      Tensor G(Bn.extent(1), des_rank);
      G.fill(gauss_rand());
      G.fill(rand());
      Tensor Y(Bn.extent(0), des_rank);
      gemm(CblasNoTrans, CblasNoTrans, 1.0, Bn, G, 0.0, Y);
      for (int j = 0; j < powerit; j++) {
        LU_decomp(Y);
        Tensor Z(Bn.extent(1), Y.extent(1));
        gemm(CblasTrans, CblasNoTrans, 1.0, Bn, Y, 0.0, Z);
        LU_decomp(Z);
        Y.resize(Range{Range1{Bn.extent(0)}, Range1{Z.extent(1)}});
        gemm(CblasNoTrans, CblasNoTrans, 1.0, Bn, Z, 0.0, Y);
      }
      QR_decomp(Y);
      transforms.push_back(Y);
      Tensor S(Y.extent(1), Y.extent(1));
      gemm(CblasTrans, CblasNoTrans, 1.0, Y, Y, 0.0, S);
      std::cout << "S" << std::endl;
      print(S);
      contract_core(Y, i);
      contract_core(Y, i, false);
      std::cout << tensor_ref << std::endl;
      print(tensor_ref);
      size = tensor_ref.size();
    }
  }
  void LU_decomp(Tensor &A) { // returns the product of the pivot and the lower
                              // triangular matrix
    btas::Tensor<int> piv(std::min(A.extent(0), A.extent(1)));
    Tensor L(A.range());
    Tensor P(A.extent(0), A.extent(0));
    P.fill(0.0);
    L.fill(0.0);
    auto info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, A.extent(0), A.extent(1),
                               A.data(), A.extent(1), piv.data());
    if (info < 0) {
      std::cout << "Error with LU decomposition" << std::endl;
      return;
    }
    if (info != 0) {
    }
    // indexing the pivod matrix
    for (auto &j : piv)
      j -= 1;
    int pivsize = piv.extent(0);
    piv.resize(Range{Range1{A.extent(0)}});
    for (int i = 0; i < piv.extent(0); i++) {
      if (i == piv(i) || i >= pivsize) {
        for (int j = 0; j < i; j++) {
          if (i == piv(j)) {
            piv(i) = j;
            break;
          }
        }
      }
      if (i >= pivsize) {
        piv(i) = i;
        for (int j = 0; j < i; j++)
          if (i == piv(j)) {
            piv(i) = j;
            break;
          }
      }
    }
    // generating the pivot matrix
    for (int i = 0; i < piv.extent(0); i++)
      P(piv(i), i) = 1;
    // generating L
    for (int i = 0; i < L.extent(0); i++) {
      for (int j = 0; j < i && j < L.extent(1); j++) {
        L(i, j) = A(i, j);
      }
      if (i < L.extent(1))
        L(i, i) = 1;
    }
    // contracting P L
    gemm(CblasNoTrans, CblasNoTrans, 1.0, P, L, 0.0, A);
  }
  void QR_decomp(Tensor &A) {
    Tensor B(1, std::min(A.extent(0), A.extent(1)));
    int Qm = A.extent(0);
    int Qn = A.extent(1);
    auto info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.extent(0), A.extent(1),
                               A.data(), A.extent(1), B.data());
    if (info == 0) {
      info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Qm, Qn, Qn, A.data(), A.extent(1),
                            B.data());
      if (info != 0) {
        std::cout << "Error in computing Q" << std::endl;
        return;
      }
    } else {
      std::cout << "Error in first lapack call QR" << std::endl;
      return;
    }
  }

  void contract_core(Tensor &Q, int mode, bool transpose = true) {
    swap_to_first(tensor_ref, mode, false, false);
    std::vector<int> temp_dims, tref_indices, Q_indicies;

    // make size of contraction for contract algorithm
    temp_dims.push_back((transpose) ? Q.extent(1) : Q.extent(0));
    for (int i = 1; i < ndim; i++)
      temp_dims.push_back(tensor_ref.extent(i));
    Tensor temp(Range{temp_dims});
    temp_dims.clear();

    // Make contraction indices
    Q_indicies.push_back((transpose) ? 0 : ndim);
    Q_indicies.push_back((transpose) ? ndim : 0);
    temp_dims.push_back(ndim);
    tref_indices.push_back(0);
    for (int i = 1; i < ndim; i++) {
      tref_indices.push_back(i);
      temp_dims.push_back(i);
    }

    contract(1.0, Q, Q_indicies, tensor_ref, tref_indices, 0.0, temp,
             temp_dims);
    tensor_ref = temp;
    size = tensor_ref.range().area();
    swap_to_first(tensor_ref, mode, true, false);
  }
  double gauss_rand() {
    std::random_device rd;
    std::mt19937 gen(1);
    std::normal_distribution<double> distribution(1.0, 1.0);
    return distribution(gen);
  }
  void print(Tensor &A) {
    if (A.rank() == 2) {
      std::cout << "{" << std::endl;
      for (int i = 0; i < A.extent(0); i++) {
        std::cout << "{" << std::endl;
        for (int j = 0; j < A.extent(1); j++)
          std::cout << A(i, j) << ",";
        std::cout << "}," << std::endl;
      }
      std::cout << "}," << std::endl;
    } else
      for (auto &i : A)
        std::cout << i << ", ";
    std::cout << std::endl;
  }
  void Tucker_compression(double epsilon_svd, std::vector<Tensor> &transforms) {
    double norm2 = norm(tensor_ref);
    norm2 *= norm2;

    for (int i = 0; i < ndim; i++) {
      auto flat = Flatten(tensor_ref, i);
      auto threshold = epsilon_svd * epsilon_svd * norm2 / ndim;
      int R = flat.extent(0);
      Tensor S(R, R), s_real(R, 1), s_image(R, 1), U(R, R), Vl(1, 1);
      s_real.fill(0.0);
      s_image.fill(0.0);
      U.fill(0.0);
      Vl.fill(0.0);

      gemm(CblasNoTrans, CblasTrans, 1.0, flat, flat, 0.0, S);
      auto info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', R, S.data(), R,
                                s_real.data(), s_image.data(), Vl.data(), R,
                                U.data(), R);
      if (info)
        return;

      int rank = 0;
      Tensor Ipiv(R, R);
      Ipiv.fill(0.0);
      // Eigenvalues are unsorted
      // This algorithm sorts the eigenvalues and generates the piviting matrix
      // for the right eigenvector which reduces the column dimension to rank
      for (int j = 0; j < R; j++) {
        auto hold = -1e10;
        int swap = 0;
        for (int k = 0; k < R; k++) {
          if (s_real(k, 0) > hold) {
            hold = s_real(k, 0);
            swap = k;
          }
        }
        s_real(swap, 0) = -1e10;
        if (hold < epsilon_svd)
          break;
        Ipiv(j, swap) = 1.0;
        rank++;
      }
      s_real = Tensor(0);
      S.resize(Range{Range1{R}, Range1{R}});
      gemm(CblasNoTrans, CblasTrans, 1.0, U, Ipiv, 0.0, S);
      auto lower_bound = {0, 0};
      auto upper_bound = {R, rank};
      auto view = btas::make_view(S.range().slice(lower_bound, upper_bound),
                                  S.storage());
      U.resize(Range{Range1{R}, Range1{rank}});
      std::copy(view.begin(), view.end(), U.begin());
      transforms.push_back(U);
      contract_core(U, i);
    }
  }
#endif //_HAS_INTEL_MKL
};

} // namespace btas
#endif // CP
