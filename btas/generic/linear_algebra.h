//
// Created by Karl Pierce on 1/26/20.
//

#ifndef BTAS_LINEAR_ALGEBRA_H
#define BTAS_LINEAR_ALGEBRA_H
#include <btas/error.h>
#include <btas/generic/lapack_extensions.h>

namespace btas{
/// Computes L of the LU decomposition of tensor \c A
/// \param[in, out] A In: A reference matrix to be LU decomposed.  Out:
/// The L of an LU decomposition of \c A.

  template <typename Tensor> void LU_decomp(Tensor &A) {

#ifndef BTAS_HAS_BLAS_LAPACK
    BTAS_EXCEPTION("LU_decomp required BLAS/LAPACK bindings to be enabled: -DBTAS_USE_BLAS_LAPACK=ON");
#else
    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;


    if (A.rank() > 2) {
      BTAS_EXCEPTION("Tensor rank > 2. Can only invert matrices.");
    }

    Tensor L(A.range());
    Tensor P(A.extent(0), A.extent(0));
    P.fill(0.0);
    L.fill(0.0);

    btas::Tensor<int64_t> piv(std::min(A.extent(0), A.extent(1)));
    auto info = getrf( blas::Layout::RowMajor, A.extent(0), A.extent(1),
                       A.data(), A.extent(1), piv.data() );
    if( info < 0) {
      BTAS_EXCEPTION("LU_decomp: GETRF had an illegal arg");
    }


    // This means that part of the LU is singular which may cause a problem in
    // ones QR decomposition but LU can be computed fine.
    if (info != 0) {
    }

    // indexing the pivot matrix
    for (auto &j : piv)
      j -= 1;

    ind_t pivsize = piv.extent(0);
    piv.resize(Range{Range1{A.extent(0)}});

    // Walk through the full pivot array and
    // put the correct index values throughout
    for (ind_t i = 0; i < piv.extent(0); i++) {
      if (i == piv(i) || i >= pivsize) {
        for (ind_t j = 0; j < i; j++) {
          if (i == piv(j)) {
            piv(i) = j;
            break;
          }
        }
      }
      if (i >= pivsize) {
        piv(i) = i;
        for (ind_t j = 0; j < i; j++)
          if (i == piv(j)) {
            piv(i) = j;
            break;
          }
      }
    }

    // generating the pivot matrix from the correct indices found above
    for (ind_t i = 0; i < piv.extent(0); i++)
      P(piv(i), i) = 1;

    // Use the output of LAPACK to make a lower triangular matrix, L
    // TODO Make this more efficient using pointer arithmetic
    for (ord_t i = 0; i < L.extent(0); i++) {
      for (ord_t j = 0; j < i && j < L.extent(1); j++) {
        L(i, j) = A(i, j);
      }
      if (i < L.extent(1))
        L(i, i) = 1;
    }

    // contracting the pivoting matrix with L to put in correct order
    gemm(blas::Op::NoTrans, blas::Op::NoTrans, 1.0, P, L, 0.0, A);
#endif
  }

/// Computes the QR decomposition of matrix \c A
/// \param[in, out] A In: A Reference matrix to be QR decomposed.  Out:
/// The Q of a QR decomposition of \c A.
/// \return bool true if QR was successful false if failed.

  template <typename Tensor> bool QR_decomp(Tensor &A) {

#ifndef BTAS_HAS_BLAS_LAPACK
    BTAS_EXCEPTION("QR_decomp required BLAS/LAPACK bindings to be enabled: -DBTAS_USE_BLAS_LAPACK=ON");
#else

    using ind_t = typename Tensor::range_type::index_type::value_type;

    if (A.rank() > 2) {
      BTAS_EXCEPTION("Tensor rank > 2. Can only QR decompose matrices.");
    }


    return !householder_qr_genq( blas::Layout::RowMajor, A.extent(0), A.extent(1),
                                 A.data(), A.extent(1) ); 
#endif
  }

/// Computes the inverse of a matrix \c A using a pivoted LU decomposition
/// \param[in, out] A In: A reference matrix to be inverted. Out:
/// The inverse of A, computed using LU decomposition.
/// \return bool true if inversion was successful false if failed
  template <typename Tensor>
  bool Inverse_Matrix(Tensor & A){

#ifndef BTAS_HAS_BLAS_LAPACK
    BTAS_EXCEPTION("INVERSE_MATRIX required BLAS/LAPACK bindings to be enabled: -DBTAS_USE_BLAS_LAPACK=ON");
#else
    if(A.rank() > 2){
      BTAS_EXCEPTION("Tensor rank > 2. Can only invert matrices.");
    }


    if( A.extent(0) != A.extent(1) ) {
      BTAS_EXCEPTION("Can only invert square matrices.");
    }

    return !lu_inverse( blas::Layout::RowMajor, A.extent(0), A.data(), A.extent(0) );
#endif
  }

/// Computes the eigenvalue decomposition of a matrix \c A and
/// \param[in, out] A In: A reference matrix to be decomposed. Out:
/// The eigenvectors of the matrix \c A.
/// \param[in, out] lambda In: An empty vector with length greater than
/// or equal to the largest mode of \c A. Out: The eigenvalues of the
///  matrix \c A
  template <typename Tensor, typename RealTensor>
  void eigenvalue_decomp(Tensor& A, RealTensor& lambda) {

#ifndef BTAS_HAS_BLAS_LAPACK
    BTAS_EXCEPTION("eigenvalue_decomp required BLAS/LAPACK bindings to be enabled: -DBTAS_USE_BLAS_LAPACK=ON");
#else

    using ind_t = typename Tensor::range_type::index_type::value_type;
    using ord_t = typename range_traits<typename Tensor::range_type>::ordinal_type;

    if (A.rank() > 2) {
      BTAS_EXCEPTION("Tensor rank > 2. Tensor A must be a matrix.");
    }
    ord_t lambda_length = lambda.size();
    ind_t smallest_mode_A = (A.extent(0) < A.extent(1) ? A.extent(0) : A.extent(1));
    if (lambda_length < smallest_mode_A) {
      lambda = Tensor(smallest_mode_A);
    }

    auto info = hereig( blas::Layout::RowMajor, lapack::Job::Vec, 
                        lapack::Uplo::Upper, smallest_mode_A, A.data(),
                        smallest_mode_A, lambda.data() );
    if (info) BTAS_EXCEPTION("Error in computing the Eigenvalue decomposition");
#endif

  }

  /// Solving Ax = B using a Cholesky decomposition
  /// \param[in, out] A In: The right-hand side of the linear equation
  /// to be inverted using Cholesky. Out:
  /// the factors L and U from the factorization A = P*L*U;
  /// the unit diagonal elements of L are not stored.
  /// \param[in, out] B In: The left-hand side of the linear equation
  /// out: The solution x = A^{-1}B
  /// \return bool true if inversion was successful false if failed.
template <typename Tensor>
bool cholesky_inverse(Tensor & A, Tensor & B) {

#ifndef BTAS_HAS_BLAS_LAPACK
    BTAS_EXCEPTION("cholesky_inverse required BLAS/LAPACK bindings to be enabled: -DBTAS_USE_BLAS_LAPACK=ON");
#else

    using ind_t = typename Tensor::range_type::index_type::value_type;
    // This method computes the inverse quickly for a square matrix
    // based on MATLAB's implementation of A / B operator.
    ind_t rank = B.extent(1);
    ind_t LDB = B.extent(0);

    // XXX DBWY Col Major? // Column major here because we are solving XA = B, not AX = B
    // But as you point out below, A is symmetric positive semi-definite so Row major should
    // give the same results
    // XXX DBWY GESV not POSV?
    return !gesv( blas::Layout::ColMajor, rank, LDB, A.data(), rank, B.data(), 
                  rank );

#endif
}

/// SVD referencing code from
/// http://www.netlib.org/lapack/explore-html/de/ddd/lapacke_8h_af31b3cb47f7cc3b9f6541303a2968c9f.html
/// Fast pseudo-inverse algorithm described in
/// https://arxiv.org/pdf/0804.4809.pdf

/// \param[in] A In: A reference to the matrix to be inverted.
/// \param[in,out] fast_pI Should a faster version of the pseudoinverse be used?
/// return if \c fast_pI was successful
/// \return \f$ A^{\dagger} \f$ The pseudoinverse of the matrix A.
template <typename Tensor>
Tensor pseudoInverse(Tensor & A, bool & fast_pI) {

#ifndef BTAS_HAS_BLAS_LAPACK
    BTAS_EXCEPTION("pseudoInverse required BLAS/LAPACK bindings to be enabled: -DBTAS_USE_BLAS_LAPACK=ON");
#else // BTAS_HAS_BLAS_LAPACK

    using ind_t = typename Tensor::range_type::index_type::value_type;
    if (A.rank() > 2) {
      BTAS_EXCEPTION("PseudoInverse can only be computed on a matrix");
    }

    ind_t row = A.extent(0), col = A.extent(1);
    auto rank = (row < col ? row : col);

    if (fast_pI) {
      Tensor temp(col, col), inv(col, row);
      // compute V^{\dag} = (A^T A) ^{-1} A^T
      gemm(blas::Op::Trans, blas::Op::NoTrans, 1.0, A, A, 0.0, temp);
      fast_pI = Inverse_Matrix(temp);
      if (fast_pI) {
        gemm(blas::Op::NoTrans, blas::Op::Trans, 1.0, temp, A, 0.0, inv);
        return inv;
      } else {
        std::cout << "Fast pseudo-inverse failed reverting to normal pseudo-inverse" << std::endl;
      }
    }
    Tensor s(Range{Range1{rank}});
    Tensor U(Range{Range1{row}, Range1{row}});
    Tensor Vt(Range{Range1{col}, Range1{col}});

    gesvd(lapack::Job::AllVec, lapack::Job::AllVec, A, s, U, Vt);

    // Inverse the Singular values with threshold 1e-13 = 0
    double lr_thresh = 1e-13;
    Tensor s_inv(Range{Range1{row}, Range1{col}});
    s_inv.fill(0.0);
    for (ind_t i = 0; i < rank; ++i) {
      if (s(i) > lr_thresh)
        s_inv(i, i) = 1 / s(i);
      else
        s_inv(i, i) = 0;
    }
    s.resize(Range{Range1{row}, Range1{col}});

    // Compute the matrix A^-1 from the inverted singular values and the U and
    // V^T provided by the SVD
    gemm(blas::Op::NoTrans, blas::Op::NoTrans, 1.0, U, s_inv, 0.0, s);
    gemm(blas::Op::NoTrans, blas::Op::NoTrans, 1.0, s, Vt, 0.0, U);

    return U;
                          
#endif
  }

} // namespace btas
#endif //BTAS_LINEAR_ALGEBRA_H
