#ifndef BTAS_RANDOMIZED_DECOMP_H
#define BTAS_RANDOMIZED_DECOMP_H

#include "core_contract.h"
#include <btas/btas.h>
#include <btas/error.h>

#include <random>
#include <stdlib.h>

#ifdef _HAS_INTEL_MKL

namespace btas {

/// \param[in,out] A In: An empty matrix of size column dimension of the nth
/// mode flattened tensor provided to the randomized compression method by the
/// desired rank of the randmoized compression method.  Out: A random matrix,
/// column drawn from a random distribution and orthogonalized
template <typename Tensor> void generate_random_metric(Tensor &A) {
  using value_type = typename Tensor::value_type;
  for (auto i = 0; i < A.extent(1); i++) {
    std::random_device rd;
    // uncomment for more randomness
    // std::mt19937 gen(rd());
    std::mt19937 gen(1.0); // comment out for more randomness.
    std::normal_distribution<value_type> distribution(0.0, 10.0);
    value_type norm = 0.0;
    for (auto j = 0; j < A.extent(0); j++) {
      A(j, i) = abs(distribution(gen));
      norm += A(j, i) * A(j, i);
    }

    norm = sqrt(norm);
    for (auto j = 0; j < A.extent(0); j++) {
      A(j, i) /= norm;
    }

    distribution.reset();
  }
  QR_decomp(A);
}

/// Calculates the randomized compression of tensor \c A.
/// <a href=https://arxiv.org/pdf/1703.09074.pdf> See reference </a>
/// \param[in, out] A In: An order-N tensor to be randomly decomposed.
/// Out: The core tensor of random decomposition \param[in, out] transforms
/// In: An empty vector.  Out: The randomized decomposition factor matrices.
/// \param[in] des_rank The rank of each mode of \c A after randomized
/// decomposition. \param[in] oversampl Oversampling added to \c
/// desired_compression_rank required to provide an optimal decomposition.
/// Default = suggested = 10. \param[in] powerit Number of power iterations, as
/// specified in the literature, to scale the spectrum of each mode. Default =
/// suggested = 2.

template <typename Tensor>
void randomized_decomposition(Tensor &A, std::vector<Tensor> &transforms,
                              int des_rank, int oversampl = 10,
                              int powerit = 2) {
  // Add the oversampling to the desired rank
  auto ndim = A.rank();
  auto rank = des_rank + oversampl;

  // Walk through all the modes of A
  for (int n = 0; n < ndim; n++) {
    // Flatten A
    auto An = flatten(A, n);

    // Make and fill the random matrix Gamma
    Tensor G(An.extent(1), rank);
    generate_random_metric(G);

    // Project The random matrix onto the flatten reference tensor
    Tensor Y(An.extent(0), rank);
    gemm(CblasNoTrans, CblasNoTrans, 1.0, An, G, 0.0, Y);

    // Start power iteration
    for (int j = 0; j < powerit; j++) {
      // Find L of an LU decomposition of the projected flattened tensor
      LU_decomp(Y);
      Tensor Z(An.extent(1), Y.extent(1));

      // Find the L of an LU decomposition of the L above (called Y) projected
      // onto the rlattened reference tensor
      gemm(CblasTrans, CblasNoTrans, 1.0, An, Y, 0.0, Z);
      LU_decomp(Z);

      // Project the second L from above (called Z) onto the flattened reference
      // tensor and start power iteration over again.
      Y.resize(Range{Range1{An.extent(0)}, Range1{Z.extent(1)}});
      gemm(CblasNoTrans, CblasNoTrans, 1.0, An, Z, 0.0, Y);
    }

    // Compute the QR from Y above.  If the QR is non-singular push it into
    // transforms and project the unitary matrix onto the reference tensor
    bool QR_good = true;
    QR_good = QR_decomp(Y);

    if (!QR_good) {
      BTAS_EXCEPTION("QR did not complete successfully due to chosen "
                     "dimension. Choose desired_compression_rank <= smallest "
                     "dimension of tensor A");
    }

    transforms.push_back(Y);
  }
  for (int n = 0; n < ndim; n++) {
    core_contract(A, transforms[n], n);
  }
}

/// Computes L of the LU decomposition of tensor \c A
/// \param[in, out] A In: A reference matrix to be LU decomposed.  Out:
/// The L of an LU decomposition of \c A.

template <typename Tensor> void LU_decomp(Tensor &A) {

  btas::Tensor<int> piv(std::min(A.extent(0), A.extent(1)));
  Tensor L(A.range());
  Tensor P(A.extent(0), A.extent(0));
  P.fill(0.0);
  L.fill(0.0);

  // LAPACKE LU decomposition gives back dense L and U to be
  // restored into lower and upper triangular form, and a pivoting
  // matrix for L
  auto info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, A.extent(0), A.extent(1),
                             A.data(), A.extent(1), piv.data());

  // This means there was a problem with the LU that must be dealt with,
  // The decomposition cannot be continued.
  if (info < 0) {
    std::cout << "Error with LU decomposition" << std::endl;
    return;
  }

  // This means that part of the LU is singular which may cause a problem in
  // ones QR decomposition but LU can be computed fine.
  if (info != 0) {
  }

  // indexing the pivot matrix
  for (auto &j : piv)
    j -= 1;

  int pivsize = piv.extent(0);
  piv.resize(Range{Range1{A.extent(0)}});

  // Walk through the full pivot array and
  // put the correct index values throughout
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

  // generating the pivot matrix from the correct indices found above
  for (int i = 0; i < piv.extent(0); i++)
    P(piv(i), i) = 1;

  // Use the output of LAPACKE to make a lower triangular matrix, L
  for (int i = 0; i < L.extent(0); i++) {
    for (int j = 0; j < i && j < L.extent(1); j++) {
      L(i, j) = A(i, j);
    }
    if (i < L.extent(1))
      L(i, i) = 1;
  }

  // contracting the pivoting matrix with L to put in correct order
  gemm(CblasNoTrans, CblasNoTrans, 1.0, P, L, 0.0, A);
}

// Computes the QR decomposition of matrix \c A
/// \param[in, out] A In: A Reference matrix to be QR decomposed.  Out:
/// The Q of a QR decomposition of \c A.

template <typename Tensor> bool QR_decomp(Tensor &A) {

  int Qm = A.extent(0);
  int Qn = A.extent(1);
  Tensor B(1, std::min(Qm, Qn));

  // LAPACKE doesn't directly calculate Q. Must first call this function to
  // generate precursors to Q
  auto info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.extent(0), A.extent(1),
                             A.data(), A.extent(1), B.data());

  if (info == 0) {
    // This function generates Q.
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Qm, Qn, Qn, A.data(), A.extent(1),
                          B.data());

    // If there was some problem generating Q, i.e. it is singular, the
    // randomized decompsoition will fail.  There is an exception thrown if
    // there is a problem to stop the randomized decomposition
    if (info != 0) {
      return false;
    }
    return true;
  } else {
    return false;
  }
}

template <typename Tensor>
bool Inverse_Matrix(Tensor & A){
  if(A.rank() > 2){
    //Return exception
  }
  btas::Tensor<int> piv(std::min(A.extent(0), A.extent(1)));
  piv.fill(0);

  // LAPACKE LU decomposition gives back dense L and U to be
  // restored into lower and upper triangular form, and a pivoting
  // matrix for L
  auto info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, A.extent(0), A.extent(1),
                             A.data(), A.extent(1), piv.data());
  if(info != 0){
    A = Tensor();
    return false;
  }
  info = LAPACKE_dgetri(CblasRowMajor, A.extent(0), A.data(), A.extent(0), piv.data());
  if(info != 0){
    A = Tensor();
    return false;
  }
  return true;
}
} // namespace btas

#endif // HAS_INTEL_MKL

#endif // BTAS_RANDOMIZED_DECOMP_H
