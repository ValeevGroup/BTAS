//
// Created by efv on 1/3/21.
//

#ifndef BTAS_FWD_H
#define BTAS_FWD_H

#include <memory> // std::allocator

namespace btas {

  template <typename _T,
      typename _Allocator = std::allocator<_T> >
  class varray;

  template <typename _T, class _Range, class _Storage>
  class Tensor;

}

#endif  // BTAS_FWD_H
