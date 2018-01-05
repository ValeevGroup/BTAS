#ifndef BTAS_RANDOMIZED
#define BTAS_RANDOMIZED
#include "core_contract.h"
#include <btas/btas.h>
#include <btas/error.h>
#include <random>
#include <stdlib.h>
#ifdef _HAS_INTEL_MKL
namespace btas {

template <typename T>
T gauss_rand() {
  std::random_device rd;
  std::mt19937 gen(1);
  std::normal_distribution<T> distribution(1.0, 1.0);
  return distribution(gen);
}

template <typename Tensor>
void randomized_decomposition(Tensor &A, int des_rank,
                              std::vector<Tensor> &transforms,
                              int oversampl = 10, int powerit = 2) {
  using value_type = typename Tensor::value_type;
  auto ndim = A.rank();
  des_rank += oversampl;
  for (int i = 0; i < ndim; i++) {
    auto Bn = flatten(A, i);
    Tensor G(Bn.extent(1), des_rank);
    G.fill(gauss_rand<value_type>());
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
    bool QR_good = true;
    QR_good = QR_decomp(Y);
    if(!QR_good){
      BTAS_EXCEPTION("QR did not complete successfully due to chosen dimension. Choose desired_compression_rank <= smallest dimension of tensor A");
    }
    transforms.push_back(Y);
    core_contract(A, Y, i);
  }
}

template <typename Tensor>
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
  // indexing the pivot matrix
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

template <typename Tensor> bool QR_decomp(Tensor &A) {
  Tensor B(1, std::min(A.extent(0), A.extent(1)));
  int Qm = A.extent(0);
  int Qn = A.extent(1);
  auto info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.extent(0), A.extent(1),
                             A.data(), A.extent(1), B.data());
  if (info == 0) {
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Qm, Qn, Qn, A.data(), A.extent(1),
                          B.data());
    if (info != 0) {
      return false;
    }
    return true;
  } else {
    return false;
  }
}

} // namespace btas
#endif // HAS_INTEL_MKL
#endif // BTAS_RANDOMIZED
