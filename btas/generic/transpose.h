#ifndef __BTAS_TRANSPOSE_H
#define __BTAS_TRANSPOSE_H 1

#include <btas/error.h>
#include <btas/generic/mkl_extensions.h>

namespace btas {

  template <typename T>
  void transpose( int64_t M, int64_t N, const T* A, int64_t LDA, 
                  T* B, int64_t LDB ) {

#ifdef BTAS_HAS_INTEL_MKL
  T one {1.0};
  omatcopy('C', 'T', M, N, one, A, LDA, B, LDB );
#else
    for( int64_t j = 0; j < N; ++j )
    for( int64_t i = 0; i < M; ++i ) {
      B[j + i*LDB] = A[i + j*LDA];
    }
#endif

  }

}

#endif
