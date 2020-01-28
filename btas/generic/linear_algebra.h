//
// Created by Karl Pierce on 1/26/20.
//

#ifndef BTAS_LINEAR_ALGEBRA_H
#define BTAS_LINEAR_ALGEBRA_H
#include <btas/error.h>

#ifdef BTAS_HAS_CBLAS

namespace btas{
  /// Computes L of the LU decomposition of tensor \c A
/// \param[in, out] A In: A reference matrix to be LU decomposed.  Out:
/// The L of an LU decomposition of \c A.

  template <typename Tensor> void LU_decomp(Tensor &A) {

#ifndef BTAS_HAS_LAPACKE
    BTAS_EXCEPTION("Using this function requires LAPACKE");
#else //BTAS_HAS_LAPACKE

    if(A.rank() > 2){
      BTAS_EXCEPTION("Tensor rank > 2. Can only invert matrices.");
    }

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
      BTAS_EXCEPTION("LU_decomp: LAPACKE_dgetrf received an invalid input parameter");
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
#endif // BTAS_HAS_LAPACKE
  }

/// Computes the QR decomposition of matrix \c A
/// \param[in, out] A In: A Reference matrix to be QR decomposed.  Out:
/// The Q of a QR decomposition of \c A.

  template <typename Tensor> bool QR_decomp(Tensor &A) {

#ifndef BTAS_HAS_LAPACKE
    BTAS_EXCEPTION("Using this function requires LAPACKE");
#else //BTAS_HAS_LAPACKE

    if(A.rank() > 2){
      BTAS_EXCEPTION("Tensor rank > 2. Can only invert matrices.");
    }

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
#endif // BTAS_HAS_LAPACKE
  }

/// Computes the inverse of a matrix \c A using a pivoted LU decomposition
/// \param[in, out] A In: A reference matrix to be inverted. Out:
/// The inverse of A, computed using LU decomposition.
  template <typename Tensor>
  bool Inverse_Matrix(Tensor & A){

#ifndef BTAS_HAS_LAPACKE
    BTAS_EXCEPTION("Using LU matrix inversion requires LAPACKE");
#else //BTAS_HAS_LAPACKE

    if(A.rank() > 2){
      BTAS_EXCEPTION("Tensor rank > 2. Can only invert matrices.");
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
#endif // BTAS_HAS_LAPACKE
  }

/// Computes the eigenvalue decomposition of a matrix \c A and
/// \param[in, out] A In: A reference matrix to be decomposed. Out:
/// The eigenvectors of the matrix \c A.
/// \param[in, out] lambda In: An empty vector with length greater than
/// or equal to the largest mode of \c A. Out: The eigenvalues of the
///  matrix \c A
  template <typename Tensor>
  void eigenvalue_decomp(Tensor & A, Tensor & lambda){
#ifndef BTAS_HAS_LAPACKE
    BTAS_EXCEPTION("Using eigenvalue decomposition requires LAPACKE");
#else //BTAS_HAS_LAPACKE
    if(A.rank() > 2){
      BTAS_EXCEPTION("Tensor rank > 2. Tensor A must be a matrix.");
    }
    auto lambda_length = lambda.size();
    auto smallest_mode_A = (A.extent(0) < A.extent(1) ? A.extent(0) : A.extent(1));
    if(lambda_length < smallest_mode_A){
      BTAS_EXCEPTION("Volume of lambda must be greater than or equal to the largest mode of A");
    }

    auto info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', smallest_mode_A,
            A.data(), smallest_mode_A, lambda.data());
    if (info) BTAS_EXCEPTION("Error in computing the SVD initial guess");
#endif // BTAS_HAS_LAPACKE
  }

  /// Solving Ax = B using a Cholesky decomposition
  /// \param[in] A The right-hand side of the linear equation
  /// to be inverted using Cholesky
  /// \param[in, out] B In: The left-hand side of the linear equation
  /// out: The solution x = A^{-1}B
template <typename Tensor>
bool cholesky_inverse(Tensor & A, Tensor & B){
#ifndef BTAS_HAS_LAPACKE
  BTAS_EXCEPTION("Cholesky inverse function requires LAPACKE");
#else //BTAS_HAS_LAPACKE
    // This method computes the inverse quickly for a square matrix
    // based on MATLAB's implementation of A / B operator.
    auto rank = B.extent(1);
    int LDB = B.extent(0);

    btas::Tensor<int, DEFAULT::range, varray<int> > piv(rank);
    piv.fill(0);
    auto info = LAPACKE_dgesv(CblasColMajor, rank, LDB, A.data(), rank, piv.data(), B.data(), rank);
    if (info == 0) {
      return true;
    }
    else{
      // If inverse fails resort to the pseudoinverse
      std::cout << "Matlab square inverse failed revert to fast inverse" << std::endl;
      return false;
    }
#endif //BTAS_HAS_LAPACKE
}

/// SVD referencing code from
/// http://www.netlib.org/lapack/explore-html/de/ddd/lapacke_8h_af31b3cb47f7cc3b9f6541303a2968c9f.html
/// Fast pseudo-inverse algorithm described in
/// https://arxiv.org/pdf/0804.4809.pdf

/// \param[in] a matrix to be inverted.
/// \return a^{\dagger} The pseudoinverse of the matrix a.
template <typename Tensor>
Tensor pseudoInverse(Tensor & a, bool & fast_pI) {
#ifndef BTAS_HAS_LAPACKE
    BTAS_EXCEPTION("Computing the pseudoinverses requires LAPACKE");
#else //BTAS_HAS_LAPACKE

    if (a.rank() > 2) {
      BTAS_EXCEPTION("PseudoInverse can only be computed on a matrix");
    }

    auto row = a.extent(0), col = a.extent(1);
    auto rank = (row < col ? row : col);

    if (fast_pI) {
      Tensor temp(col, col), inv(col, row);
      // compute V^{\dag} = (A^T A) ^{-1} A^T
      gemm(CblasTrans, CblasNoTrans, 1.0, a, a, 0.0, temp);
      fast_pI = Inverse_Matrix(temp);
      if (fast_pI) {
        gemm(CblasNoTrans, CblasTrans, 1.0, temp, a, 0.0, inv);
        return inv;
      } else {
        std::cout << "Fast pseudo-inverse failed reverting to normal pseudo-inverse" << std::endl;
      }
    }
    Tensor s(Range{Range1{rank}});
    Tensor U(Range{Range1{row}, Range1{row}});
    Tensor Vt(Range{Range1{col}, Range1{col}});

    gesvd('A', 'A', a, s, U, Vt);

    // Inverse the Singular values with threshold 1e-13 = 0
    double lr_thresh = 1e-13;
    Tensor s_inv(Range{Range1{row}, Range1{col}});
    s_inv.fill(0.0);
    for (auto i = 0; i < rank; ++i) {
      if (s(i) > lr_thresh)
        s_inv(i, i) = 1 / s(i);
      else
        s_inv(i, i) = s(i);
    }
    s.resize(Range{Range1{row}, Range1{col}});

    // Compute the matrix A^-1 from the inverted singular values and the U and
    // V^T provided by the SVD
    gemm(CblasNoTrans, CblasNoTrans, 1.0, U, s_inv, 0.0, s);
    gemm(CblasNoTrans, CblasNoTrans, 1.0, s, Vt, 0.0, U);

    return U;
#endif // BTAS_HAS_LAPACKE
  }

} // namespace btas
#endif // BTAS_HAS_CBLAS
#endif //BTAS_LINEAR_ALGEBRA_H
