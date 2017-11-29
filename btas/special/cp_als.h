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
      : tensor_ref(tensor), dim(tensor_ref.rank()), size(tensor_ref.size()) {

#if not defined(BTAS_HAS_CBLAS) || not defined(_HAS_INTEL_MKL)
    BTAS_EXCEPTION_MESSAGE(__FILE__, __LINE__,
                           "CP_ALS requires LAPACKE or mkl_lapack");
#endif
#ifdef _HAS_INTEL_MKL
#include <mkl_trans.h>
#endif
    for (int i = 0; i < dim; i++)
      curr_dim.push_back(tensor_ref.extent(i));

    oldmat.resize(Range{Range1{tensor_ref.extent(0)},
                        Range1{size / tensor_ref.extent(0)}});
  }

  ~CP_ALS() = default;

  //Builds rank R approximation starting from r=1 to R
  //direct method doesn't compute Khatri-Rao intermediate
  //r_test ||T(0) - U(0)||F where U(1) is the recomposed
    //approximate tensor and  T(1) is the tensor flattened
    //along mode 0. Not required for finite rank decomposition
  //Max_R is the max rank, used by finite error computation
  //skip allows rank R approximation to be built from skip
    //many columns at a time
  //ep_ALS is the threshold for the ALS minimization
  //epsilon is the threshold for the CP decomposition

  void compute(const int rank, const bool direct = true,
               const bool r_test = false, const double max_R = 1e5,
               const int skip = 1, const double ep_ALS = 0.1) {
    Build(rank, direct, max_R, r_test, skip, ep_ALS);
  }

  void compute(const double epsilon = 1e-2, const bool direct = true,
               const double max_R = 1e5, const int skip = 1,
               const double ep_ALS = 0.1) {
    int rank = (A.empty()) ? 0 : A[0].extent(0);
    while (rank_test > epsilon && rank < max_R) {
      rank += skip;
      compute(rank, direct, true, max_R, skip, ep_ALS);
      std::cout << "With rank " << rank << " there is a difference of "
                << rank_test << std::endl;
    }
    std::cout << "The decomposition finished with rank " << rank
              << " and epsilon " << rank_test << std::endl;
  }
  //computes ||T(0) - U(0)||F and returns the result
  double difference() {
    std::cout.precision(16);
    auto rank = A[0].extent(1);
    double diff = 0;
    Tensor oldmat(tensor_ref.extent(0), size / tensor_ref.extent(0));
    for (int i = 0; i < rank; i++) {
      scal(A[0].extent(0), A[dim](i), std::begin(A[0]) + i, rank);
    }
    gemm(CblasNoTrans, CblasTrans, 1.0, A[0], generate_KRP(0, rank, true),
         0.0, oldmat);
    for (int i = 0; i < rank; i++) {
      scal(A[0].extent(0), 1 / A[dim](i), std::begin(A[0]) + i, rank);
    }
    diff = std::abs(norm(Flatten(0) - oldmat));
    return diff;
  }
  //Vector of (number of modes) factor matrices
  //plus an array of weights, stored last.
  std::vector<Tensor> A;  

private:
  ///////Global variables//////////

  Tensor &tensor_ref; // this is a reference.
  const int dim;
  int size;
  Tensor oldmat;
  double test = 1.0;
  std::vector<size_t> curr_dim; // Need this in LD_Switch because tensor_ref
                                // range object cannot handle swaps easily

  double rank_test = 10.0;
  // This is the rank r residual between tensor_ref and A

  ///////CP-ALS Specific functions to build and optimize guesses /////////

  // creates factor matricies starting with R=1 and moves to R = rank
  // Where R is the column dimension of the factor matrices.
  void Build(const int rank, const bool dir, const int max_R, const bool r_test,
             const int skip, const double ep_ALS) {
    std::cout.precision(15);
    for (auto i = (A.empty()) ? 0 : A.at(0).extent(1); i < rank; i += skip) {
      for (auto j = 0; j < dim; ++j) {
        if (i == 0) { // creates a factor matrix when A is empty
          Tensor a(Range{tensor_ref.range(j), Range1{i + 1}});
          a.fill(rand());
          normCol(a, i);
          A.push_back(a);
          if (j + 1 == dim) {
            Tensor lam(Range{Range1{i + 1}});
            A.push_back(lam);
          }
        } else { // rebuilds factor matrix with column dimension increased by skip
          Tensor b(Range{A[0].range(0), Range1{i + 1}});
          b.fill(rand());
          for (int l = A[0].extent(1); l < i + 1; ++l)
            normCol(b, l);
          for (int k = 0; k < b.extent(0); k++)
            for (int l = 0; l < A[0].extent(1); l++)
              b(k, l) = A[0](k, l);

          A.erase(A.begin());
          A.push_back(b);
          if (j + 1 == dim) {
            b.resize(Range{Range1{i + 1}});
            for (int k = 0; k < A[0].extent(0); k++)
              b(k) = A[0](k);
            A.erase(A.begin());
            A.push_back(b);
          }
        }
      }
      ALS(i + 1, dir, max_R, r_test,
          ep_ALS); // performs the least squares minimization to generate the
                   // best CP rank i+1 approximation
    }
  }

  // performs the ALS method to minimize the loss function for a single rank
  void ALS(const int rank, const bool dir, const int max_R, const bool r_test,
           const double ep_ALS) {
    auto count = 0;
    test = 1.0;
    while (count <= max_R && test > ep_ALS) {
      count++;
      test = 0.0;
      for (auto i = 0; i < dim; i++) {
        if (dir)
          direct(i, rank); //Does not compute the Khatri-Rao product
        else
          update_w_KRP(i, rank); 
      }
    }
    // only checks loss function if required
    if (r_test) {
      for (int i = 0; i < rank; i++) {
        scal(A[0].extent(0), A[dim](i), std::begin(A[0]) + i, rank);
      }
      gemm(CblasNoTrans, CblasTrans, 1.0, A[0], generate_KRP(0, rank, true),
           0.0, oldmat);
      for (int i = 0; i < rank; i++) {
        scal(A[0].extent(0), 1 / A[dim](i), std::begin(A[0]) + i, rank);
      }
      rank_test = std::abs(norm(Flatten(0) - oldmat));
    }
  }

  void update_w_KRP(int n, int rank) {
    Tensor temp(A[n].extent(0), rank);
    Tensor an(A[n].range());
#ifdef _HAS_INTEL_MKL
    //with MKL no flattening required
    LD_Switch(n, 'b');
    auto KhatriRao = generate_KRP(n, rank, false);
    contract_w_krp(KhatriRao, rank, temp);
    LD_Switch(n, 'f');

#else // BTAS_HAS_CBLAS
    //Without MKL flattening required
    gemm(CblasNoTrans, CblasNoTrans, 1.0, Flatten(n),
         generate_KRP(n, rank, true), 0.0, temp);
#endif
    gemm(CblasNoTrans, CblasNoTrans, 1.0, temp, pseudoInverse(n, rank), 0.0,
         an);
    //normalizes the updated factor matrix and compares it to the previous
    //iterations optimization
    //this optimization doesn't require recomputing of the flattened matrix 
    for (auto l = 0; l < rank; ++l)
      A[dim](l) = normCol(an, l);
    test += norm(A[n] - an);
    A[n] = an;
  }
  void direct(const int n, const int rank) {
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
    int LH_size = size;
    int contract_dim = dim - 1;
    int pseudo_rank = rank;
    Tensor temp(1, 1);
    Range R = tensor_ref.range();
    Tensor an(A[n].range());
    if (n < dim - 1) {
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
      while (contract_dim < dim - 1) {
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
      A[dim](l) = normCol(an, l);
    test += norm(A[n] - an);
    A[n] = an;
  }

  void contract_w_krp(Tensor &KhatriRao, int rank, Tensor &product,
                      bool first_index = true) {
    //contraction of tensor objects of different dimension, currently general implementation
    enum { i, j, k, l, m, n, o, p, r };
    Range range = tensor_ref.range();
    if (first_index) {
      if (dim == 3) {
        KhatriRao.resize(
            Range{Range1{curr_dim[1]}, Range1{curr_dim[2]}, Range1{rank}});
        tensor_ref.resize(Range{Range1{curr_dim[0]}, Range1{curr_dim[1]},
                                Range1{curr_dim[2]}});
        contract(1.0, tensor_ref, {i, j, k}, KhatriRao, {j, k, r}, 0.0, product,
                 {i, r});
      } else if (dim == 4) {
        KhatriRao.resize(Range{Range1{curr_dim[1]}, Range1{curr_dim[2]},
                               Range1{curr_dim[3]}, Range1{rank}});
        tensor_ref.resize(Range{Range1{curr_dim[0]}, Range1{curr_dim[1]},
                                Range1{curr_dim[2]}, Range1{curr_dim[3]}});
        contract(1.0, tensor_ref, {i, j, k, l}, KhatriRao, {j, k, l, r}, 0.0,
                 product, {i, r});
      } else if (dim == 5) {
        KhatriRao.resize(Range{Range1{curr_dim[1]}, Range1{curr_dim[2]},
                               Range1{curr_dim[3]}, Range1{curr_dim[4]},
                               Range1{rank}});
        tensor_ref.resize(Range{Range1{curr_dim[0]}, Range1{curr_dim[1]},
                                Range1{curr_dim[2]}, Range1{curr_dim[3]},
                                Range1{curr_dim[4]}});
        contract(1.0, tensor_ref, {i, j, k, l, m}, KhatriRao, {j, k, l, m, r},
                 0.0, product, {i, r});
      } else if (dim == 6) {
        KhatriRao.resize(Range{Range1{curr_dim[1]}, Range1{curr_dim[2]},
                               Range1{curr_dim[3]}, Range1{curr_dim[4]},
                               Range1{curr_dim[5]}, Range1{rank}});
        tensor_ref.resize(Range{Range1{curr_dim[0]}, Range1{curr_dim[1]},
                                Range1{curr_dim[2]}, Range1{curr_dim[3]},
                                Range1{curr_dim[4]}, Range1{curr_dim[5]}});
        contract(1.0, tensor_ref, {i, j, k, l, m, n}, KhatriRao,
                 {j, k, l, m, n, r}, 0.0, product, {i, r});
      } else if (dim == 7) {
        KhatriRao.resize(Range{Range1{curr_dim[1]}, Range1{curr_dim[2]},
                               Range1{curr_dim[3]}, Range1{curr_dim[4]},
                               Range1{curr_dim[5]}, Range1{curr_dim[6]},
                               Range1{rank}});
        tensor_ref.resize(Range{Range1{curr_dim[0]}, Range1{curr_dim[1]},
                                Range1{curr_dim[2]}, Range1{curr_dim[3]},
                                Range1{curr_dim[4]}, Range1{curr_dim[5]},
                                Range1{curr_dim[6]}});
        contract(1.0, tensor_ref, {i, j, k, l, m, n, o}, KhatriRao,
                 {j, k, l, m, n, o, r}, 0.0, product, {i, r});
      } else if (dim == 8) {
        KhatriRao.resize(Range{Range1{curr_dim[1]}, Range1{curr_dim[2]},
                               Range1{curr_dim[3]}, Range1{curr_dim[4]},
                               Range1{curr_dim[5]}, Range1{curr_dim[6]},
                               Range1{curr_dim[7]}, Range1{rank}});
        tensor_ref.resize(Range{Range1{curr_dim[0]}, Range1{curr_dim[1]},
                                Range1{curr_dim[2]}, Range1{curr_dim[3]},
                                Range1{curr_dim[4]}, Range1{curr_dim[5]},
                                Range1{curr_dim[6]}, Range1{curr_dim[7]}});
        contract(1.0, tensor_ref, {i, j, k, l, m, n, o}, KhatriRao,
                 {j, k, l, m, n, o, r}, 0.0, product, {i, r});
      } else {
        std::stringstream ss;
        ss << "not yet implemented: dimension " << dim;
        throw std::logic_error(ss.str());
      }
      tensor_ref.resize(range);
    } else {
      std::cout << "curr_dim" << std::endl;
      for (auto &i : curr_dim)
        std::cout << i << std::endl;
      if (dim == 3) {
        KhatriRao.resize(Range{Range1{A[curr_dim[0]].extent(0)},
                               Range1{A[curr_dim[1]].extent(0)}, Range1{rank}});
        tensor_ref.resize(Range{Range1{A[curr_dim[0]].extent(0)},
                                Range1{A[curr_dim[1]].extent(0)},
                                Range1{A[curr_dim[2]].extent(0)}});
        for (int a = 0; a < KhatriRao.extent(0); a++)
          for (int b = 0; b < KhatriRao.extent(1); b++)
            for (int c = 0; c < tensor_ref.extent(2); c++)
              for (int d = 0; d < KhatriRao.extent(2); d++)
                product(c, d) += tensor_ref(a, b, c) * KhatriRao(a, b, d);
      } else if (dim == 4) {
        KhatriRao.resize(Range{Range1{curr_dim[0]}, Range1{curr_dim[1]},
                               Range1{curr_dim[2]}, Range1{rank}});
        tensor_ref.resize(Range{Range1{curr_dim[0]}, Range1{curr_dim[1]},
                                Range1{curr_dim[2]}, Range1{curr_dim[3]}});
        contract(1.0, tensor_ref, {i, j, k, l}, KhatriRao, {i, j, k, r}, 0.0,
                 product, {l, r});
      } else {
        std::stringstream ss;
        ss << "not yet implemented: dimension " << dim;
        throw std::logic_error(ss.str());
      }
      tensor_ref.resize(range);
    }
  }

  Tensor generate_V(const int n, const int rank) {
    // To generate V first Multiply A^T.A then Hadamard product V(i,j) *=
    // A^T.A(i,j);
    Tensor V(rank, rank);
    V.fill(1.0);
    for (auto j = 0; j < dim; ++j) {
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
    Tensor temp(Range{Range1{A.at(n).extent(0)}, Range1{rank}});
    Tensor left_side_product(Range{Range1{rank}, Range1{rank}});
    if (!forward) { // forward direction
      for (auto i = 0; i < dim; ++i) {
        if ((i == 0 && n != 0) || (i == 1 && n == 0)) {
          left_side_product = A.at(i);
        } else if (i != n) {
          KhatriRaoProduct(left_side_product, A[i], temp);
          left_side_product = temp;
        }
      }
    } else { // backward direction
      for (auto i = dim - 1; i > -1; --i) {
        if ((i == dim - 1 && n != dim - 1) || (i == dim - 2 && n == dim - 1)) {
          left_side_product = A.at(i);
        }

        else if (i != n) {
          KhatriRaoProduct(left_side_product, A[i], temp);
          left_side_product = temp;
        }
      }
    }
    return left_side_product;
  }

  void KhatriRaoProduct(
      const Tensor &B, const Tensor &C,
      Tensor &BC_product) { 
    // The khatri-rao product is an outer product of column vectors in
    // two matrices, then ordered to make a super column in a new matrix
    // The dimension of this product is  B(NXM) (.) C(KXM) = D(N*K X M)
    BC_product.resize(
        Range{Range1{B.extent(0) * C.extent(0)}, Range1{B.extent(1)}});
    for (auto i = 0; i < B.extent(0); ++i)
      for (auto j = 0; j < C.extent(0); ++j)
        for (auto k = 0; k < B.extent(1); ++k)
          BC_product(i * C.extent(0) + j, k) = B(i, k) * C(j, k);
  }

#ifdef _HAS_INTEL_MKL
  void LD_Switch(int d, char FB = 'b') {
    // For switching when the dimension of interests is in the back use b/B
    // otherwise use f/F
    // A{T1,T2,T3,T4,T5} (3,b) -> A{T3,T1,T2,T4,T5}
    // A{T3, T1, T2, T4, T5} (3, f) -> A{T1, T2, T3, T4, T5}
    FB = (FB == 'B') ? 'b' : FB;
    FB = (FB == 'F') ? 'f' : FB;
    if (FB != 'b' && FB != 'f')
      BTAS_EXCEPTION_MESSAGE(
          __FILE__, __LINE__,
          "Invalid dimension swapping directive. Requires 'f' or 'b'");
    size_t rows = 1;
    size_t cols = 1;
    auto step = 1;
    if (d == 0)
      return;
    else if (d == dim - 1) {
      rows = size / tensor_ref.extent(d);
      cols = tensor_ref.extent(d);
      double *data_ptr = tensor_ref.data();
      if (FB == 'b')
        mkl_dimatcopy('R', 'T', rows, cols, 1.0, data_ptr, cols, rows);
      else
        mkl_dimatcopy('R', 'T', cols, rows, 1.0, data_ptr, rows, cols);

      step = curr_dim[0];
      curr_dim[0] = curr_dim[d];
      curr_dim[d] = step;
    } else {
      for (int i = 0; i <= d; i++)
        rows *= tensor_ref.extent(i);
      cols = size / rows;
      double *data_ptr = tensor_ref.data();
      mkl_dimatcopy('R', 'T', rows, cols, 1.0, data_ptr, cols, rows);
      step = rows;
      for (int i = 0; i < cols; i++) {
        size_t in_rows = rows / tensor_ref.extent(d);
        size_t in_cols = tensor_ref.extent(d);
        data_ptr = tensor_ref.data() + i * step;
        if (FB == 'b')
          mkl_dimatcopy('R', 'T', in_rows, in_cols, 1.0, data_ptr, in_cols,
                        in_rows);
        else
          mkl_dimatcopy('R', 'T', in_cols, in_rows, 1.0, data_ptr, in_rows,
                        in_cols);
      }
      data_ptr = tensor_ref.data();
      mkl_dimatcopy('R', 'T', cols, rows, 1.0, data_ptr, rows, cols);
      step = curr_dim[0];
      curr_dim[0] = curr_dim[d];
      curr_dim[d] = step;
    }
  }
#endif

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
    gesvd(CblasRowMajor, 'A', 'A', a.extent(0), a.extent(1), std::begin(a),
          a.extent(1), std::begin(s), std::begin(U), a.extent(0),
          std::begin(Vt), a.extent(1));
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

  // two methods to produce to matricize a tensor along the n-th fiber
  Tensor Flatten(int n) {
    Tensor X(Range{Range1{tensor_ref.extent(n)},
                   Range1{size / tensor_ref.extent(n)}});
    X.fill(1.0);
    int indexi = 0, indexj = 0;
    std::vector<int> J(dim, 1);
    for (auto i = 0; i < dim; ++i)
      if (i != n)
        for (auto m = 0; m < i; ++m)
          if (m != n)
            J[i] *= tensor_ref.extent(m);

    iterator tensor_itr = tensor_ref.begin();
    fill(0, X, n, indexi, indexj, J, tensor_itr);
    return X;
  }
  //recursivly calls fill to calculate the correct column step
  //size for the matrix flattened on the n-th fiber
  void fill(int depth, Tensor &X, int n, int indexi, int indexj,
            std::vector<int> &J, iterator &tensor_itr) {
    if (depth < dim) {
      for (auto i = 0; i < tensor_ref.extent(depth); ++i) {
        if (depth != n) {
          indexj += i * J[depth]; // column matrix index
        } else {
          indexi = i; // row matrix index
        }
        fill(depth + 1, X, n, indexi, indexj, J, tensor_itr);
        if (depth != n)
          indexj -= i * J[depth];
      }
    } else {
      X(indexi, indexj) = *tensor_itr;
      tensor_itr++;
    }
  }
};

} // namespace btas
#endif // CP
