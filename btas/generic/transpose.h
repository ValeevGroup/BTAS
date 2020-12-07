#ifndef __BTAS_TRANSPOSE_H
#define __BTAS_TRANSPOSE_H 1

#include <btas/error.h>
#include <btas/generic/mkl_extensions.hpp>

namespace btas {

  template <typename T>
  void transpose( int64_t M, int64_t N, const T* A, int64_t LDA, 
                  T* B, int64_t LDB ) {

  std::cout << "IN TRANSPOSE" << std::endl;
  std::cout << BTAS_HAS_INTEL_MKL << std::endl;

#ifdef BTAS_HAS_INTEL_MKL
  omatcopy( 'C', 'T', M, N, T(1.), A, LDA, B, LDB );
#else
    for( int64_t j = 0; j < N; ++j )
    for( int64_t i = 0; i < M; ++i ) {
      B[j + i*LDB] = A[i + j*LDA];
    }
#endif

  }

}

#endif
