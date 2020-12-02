#ifndef __BTAS_TRANSPOSE_H
#define __BTAS_TRANSPOSE_H 1

namespace btas {

  template <typename T>
  void transpose( int64_t M, int64_t N, const T* A, int64_t LDA, 
                  T* B, int64_t LDB ) {

    for( int64_t j = 0; j < N; ++j )
    for( int64_t i = 0; i < M; ++i ) {
      B[j + i*LDB] = A[i + j*LDA];
    }

  }

}

#endif
