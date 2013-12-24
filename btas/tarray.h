#ifndef __BTAS_TARRAY_H
#define __BTAS_TARRAY_H 1

#include <array>
#include <vector>
#include <algorithm>
#include <type_traits>

#include <btas/types.h>
#include <btas/defaults.h>
#include <btas/tensor_traits.h>
#include <btas/range.h>

#include <btas/util/stride.h>
#include <btas/util/dot.h>

namespace btas {

  /// Fixed-rank version of TArray
  template<typename _T,
           unsigned long _N,
           CBLAS_ORDER _Order = CblasRowMajor,
           class _Container = DEFAULT::storage<_T>>
  using TArray = Tensor<_T, RangeNd<_Order, std::array<long, _N> >, _Container >;

} // namespace btas

#endif // __BTAS_TARRAY_H
