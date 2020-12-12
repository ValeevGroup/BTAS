#ifndef __BTAS_TYPES_H
#define __BTAS_TYPES_H 1

//
//  BLAS types
//

#include <complex>

#ifdef BTAS_HAS_BLAS_LAPACK

#include <blas.hh>
#include <lapack.hh>

#else

namespace blas {

enum class Layout : char {
  RowMajor = 'R',
  ColMajor = 'C'
};

enum class Op : char {
  NoTrans   = 'N',
  Trans     = 'T',
  ConjTrans = 'C'
};

enum class Uplo : char {
  Upper = 'U',
  Lower = 'L'
};

}

namespace lapack {

enum class Job : char {
  Vec          = 'V',
  NoVec        = 'N', 
  AllVec       = 'A',
  OverwriteVec = 'O'
};

typedef blas::Uplo Uplo;


}

#endif

namespace btas {

  //
  //  Other aliases for convenience
  //

  /// default size type
  typedef unsigned long size_type;

  /// null deleter
  struct nulldeleter {
    void operator()(void const*) {}
  };

}

#endif // __BTAS_TYPES_H
