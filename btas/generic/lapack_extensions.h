//
// Created by David Williams-Young on 12/2/20.
//

#ifndef BTAS_LAPACK_EXTENSIONS_H
#define BTAS_LAPACK_EXTENSIONS_H 1

#ifdef BTAS_HAS_BLAS_LAPACK

#include <blas.hh>   // BLASPP
#include <lapack.hh> // LAPACKPP
#include <btas/generic/transpose.h> // Transpose
#include <btas/type_traits.h>

namespace btas {

template <typename T, typename Alloc = std::allocator<T>>
int64_t getrf( blas::Layout order, int64_t M, int64_t N, T* A, int64_t LDA,
               int64_t* IPIV, Alloc alloc = Alloc() ) {

  //std::cout << "IN GETRF IMPL" << std::endl;
  if( order == blas::Layout::ColMajor ) {
    return lapack::getrf( M, N, A, LDA, IPIV );
  } else {

    // Transpose input
    auto* A_transpose = alloc.allocate(M*N);
    transpose( N, M, A, LDA, A_transpose, M );

    // A -> LU
    auto info = lapack::getrf( M, N, A_transpose, M, IPIV );

    // Transpose output + cleanup
    if(!info)
      transpose( M, N, A_transpose, M, A, LDA );
    alloc.deallocate( A_transpose, M*N );

    return info;
  }

}

template <typename T, typename Alloc = std::allocator<T>>
int64_t geqp3_pivot( blas::Layout order, int64_t M, int64_t N, T* A, int64_t LDA,
              int64_t* IPIV, T* tau, Alloc alloc = Alloc() ) {

  //std::cout << "IN GETRF IMPL" << std::endl;
  if( order == blas::Layout::ColMajor ) {
    return lapack::geqp3( M, N, A, LDA, IPIV, tau);
  } else {

    // Transpose input
    auto* A_transpose = alloc.allocate(M*N);
    transpose( N, M, A, LDA, A_transpose, M );

    // A -> LU
    auto info = lapack::geqp3( M, N, A_transpose, M, IPIV, tau);

    // Transpose output + cleanup
    if(!info)
      transpose( M, N, A_transpose, M, A, LDA );
    alloc.deallocate( A_transpose, M*N );

    return info;
  }

}

template <typename T, typename Alloc = std::allocator<T>, 
          typename IntAlloc = std::allocator<int64_t> >
int64_t gesv( blas::Layout order, int64_t N, int64_t NRHS, T* A, int64_t LDA,
              T* B, int64_t LDB, Alloc alloc = Alloc(),
              IntAlloc int_alloc = IntAlloc() ) {

  //std::cout << "IN GESV IMPL" << std::endl;
  // Allocate IPIV
  auto* IPIV = int_alloc.allocate(N);

  auto* A_use = A;
  int64_t LDA_use = LDA;

  // If row major, transpose input and redirect pointers
  if( order == blas::Layout::RowMajor ) {
    A_use   = alloc.allocate( N*N );
    LDA_use = N;
    transpose( N, N, A, LDA, A_use, N );
  }

  auto info = lapack::gesv( N, NRHS, A, LDA, IPIV, B, LDB );

  int_alloc.deallocate( IPIV, N );

  // If row major, transpose output + cleanup
  if( order == blas::Layout::RowMajor ) {
    if(!info) transpose( N, N, A_use, N, A, LDA );
    alloc.deallocate( A_use, N*N );
  }

  return info;
}


template <typename T>
int64_t gesvd( blas::Layout order, lapack::Job jobu, lapack::Job jobvt,
               int64_t M, int64_t N, T* A, int64_t LDA, real_type_t<T>* S, 
               T* U, int64_t LDU, T* VT, int64_t LDVT ) {

  //std::cout << "IN GESVD IMPL" << std::endl;
  // Col major, no changes
  if( order == blas::Layout::ColMajor ) {

    return lapack::gesvd( jobu, jobvt, M, N, A, LDA, S, U, LDU, VT, LDVT );

  // Row major, swap M <-> N and U <-> VT
  } else {

    return lapack::gesvd( jobvt, jobu, N, M, A, LDA, S, VT, LDVT, U, LDU );

  }

}


template <typename T, typename Alloc = std::allocator<T>>
int64_t householder_qr_genq( blas::Layout order, int64_t M, int64_t N, T* A,
                             int64_t LDA, Alloc alloc = Alloc() ) {

  //std::cout << "IN QR IMPL" << std::endl;
  // Allocate temp storage for TAU factors
  const int64_t K = std::min(M,N);
  auto* TAU = alloc.allocate( K );
  int64_t info;


  auto* A_use = A;
  int64_t LDA_use = LDA;

  // If row major, transpose input and redirect pointers
  if( order == blas::Layout::RowMajor ) {
    A_use   = alloc.allocate( M*N );
    LDA_use = M;
    transpose( N, M, A, LDA, A_use, M );
  }


  // Generate QR factors in El reflector form
  info = lapack::geqrf( M, N, A_use, LDA_use, TAU );
  if( !info ) {
  
    // Generate Q from reflectors

    // Real -> XORGQR
    if constexpr ( not is_complex_type_v<T> ) 
      info = lapack::orgqr( M, N, K, A_use, LDA_use, TAU );
    // Complex -> XUNGQR 
    else
      info = lapack::ungqr( M, N, K, A_use, LDA_use, TAU );

  }

  // If row major, transpose output + cleanup
  if( order == blas::Layout::RowMajor ) {
    if(!info) transpose( M, N, A_use, M, A, LDA );
    alloc.deallocate( A_use, M*N );
  }

  // Cleanup Tau
  alloc.deallocate( TAU, K );

  return info;
}

template <typename T, typename Alloc = std::allocator<T>, 
          typename IntAlloc = std::allocator<int64_t> >
int64_t lu_inverse( blas::Layout order, int64_t N, T* A, int64_t LDA,
                    Alloc alloc = Alloc(), 
                    IntAlloc int_alloc = IntAlloc() ) {

  //std::cout << "IN LU INV IMPL" << std::endl;
  auto* A_use = A;
  int64_t LDA_use = LDA;

  // If row major, transpose input and redirect pointers
  if( order == blas::Layout::RowMajor ) {
    A_use   = alloc.allocate( N*N );
    LDA_use = N;
    transpose( N, N, A, LDA, A_use, N );
  }

  // Allocate Pivot
  int64_t* IPIV = int_alloc.allocate( N );


  // A -> LU
  int64_t info = lapack::getrf( N, N, A_use, LDA_use, IPIV );

  // Generate inverse
  if( !info ) {
    info = lapack::getri( N, A_use, LDA_use, IPIV );
  }

  // If row major + sucessful, transpose output + cleanup
  if( order == blas::Layout::RowMajor ) {
    if(!info) transpose( N, N, A_use, N, A, LDA );
    alloc.deallocate( A_use, N*N );
  }

  // Cleanup Pivot
  int_alloc.deallocate( IPIV, N );

  return info;
}


template <typename T, typename Alloc = std::allocator<T>>
int64_t hereig( blas::Layout order, lapack::Job jobz, lapack::Uplo uplo, 
                int64_t N, T* A, int64_t LDA, real_type_t<T>* W,
                Alloc alloc = Alloc() ) {

  //std::cout << "IN HEREIG IMPL" << std::endl;
  // If row major, Swap uplo
  if( order == blas::Layout::RowMajor ) {

    if( uplo == lapack::Uplo::Lower ) uplo = lapack::Uplo::Upper;
    else                              uplo = lapack::Uplo::Lower;

  }


  int64_t info;

  // Complex -> XHEEV
  if constexpr (is_complex_type_v<T>)
    info = lapack::heev( jobz, uplo, N, A, LDA, W );
  else
    info = lapack::syev( jobz, uplo, N, A, LDA, W );

  // If row major + sucessful + vectors wanted, transpose output
  if( !info and  order == blas::Layout::RowMajor and jobz == lapack::Job::Vec ) {

    // Allocate scratch space
    auto* A_t = alloc.allocate(N*N);
    transpose( N, N, A, LDA, A_t, N );
    
    // If complex, conjugate (A**T = CONJ(A))
    if constexpr ( is_complex_type_v<T> ) {
      for( int64_t i = 0; i < N*N; ++i ) A_t[i] = std::conj(A_t[i]);
    }

    // Copy back to output vars
    for( int64_t i = 0; i < N; ++i )
    for( int64_t j = 0; j < N; ++j )
      A[i*LDA + j] = A_t[i*N + j];
  
    // Free scratch
    alloc.deallocate( A_t, N*N );

  }
  
  return info;
}

}

#endif // BLAS_LAPACK

#endif
