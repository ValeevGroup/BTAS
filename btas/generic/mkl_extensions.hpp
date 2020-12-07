#ifndef BTAS_MKL_EXTENSIONS_H
#define BTAS_MKL_EXTENSIONS_H

#ifdef BTAS_HAS_INTEL_MKL

#include <mkl_trans.h>
#include <btas/type_traits.h>

namespace btas {

template <typename T, typename = std::enable_if_t< is_blas_lapack_type_v<T> >>
void imatcopy( char ORDERING, char TRANS, MKL_INT M, MKL_INT N, 
               std::type_identity_t<T> SCAL, T* A, MKL_INT SRC_LDA,
               MKL_INT DST_LDA ) {

  if constexpr ( std::is_same_v<T, float > )
    mkl_simatcopy( ORDERING, TRANS, M, N, T(SCAL), A, SRC_LDA, DST_LDA );
  else if constexpr ( std::is_same_v<T, double > )
    mkl_dimatcopy( ORDERING, TRANS, M, N, T(SCAL), A, SRC_LDA, DST_LDA );
  else if constexpr ( std::is_same_v<T, std::complex<float> > )
    mkl_cimatcopy( ORDERING, TRANS, M, N, T(SCAL), A, SRC_LDA, DST_LDA );
  else if constexpr ( std::is_same_v<T, std::complex<double> > )
    mkl_zimatcopy( ORDERING, TRANS, M, N, T(SCAL), A, SRC_LDA, DST_LDA );
  else
    BTAS_EXCEPTION( "Somehow made it into an unsupported IMATCOPY path" );

}

template <typename T, typename = std::enable_if_t< is_blas_lapack_type_v<T> >>
void omatcopy( char ORDERING, char TRANS, MKL_INT M, MKL_INT N, 
               std::type_identity_t<T> SCAL, const T* A, MKL_INT LDA,
               T* B, MKL_INT LDB ) {

  if constexpr ( std::is_same_v<T, float > )
    mkl_somatcopy( ORDERING, TRANS, M, N, T(SCAL), A, LDA, B, LDB );
  else if constexpr ( std::is_same_v<T, double > )
    mkl_domatcopy( ORDERING, TRANS, M, N, T(SCAL), A, LDA, B, LDB );
  else if constexpr ( std::is_same_v<T, std::complex<float> > )
    mkl_comatcopy( ORDERING, TRANS, M, N, T(SCAL), A, LDA, B, LDB );
  else if constexpr ( std::is_same_v<T, std::complex<double> > )
    mkl_zomatcopy( ORDERING, TRANS, M, N, T(SCAL), A, LDA, B, LDB );
  else
    BTAS_EXCEPTION( "Somehow made it into an unsupported OMATCOPY path" );

}

}

#endif

#endif
