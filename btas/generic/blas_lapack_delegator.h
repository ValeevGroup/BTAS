#pragma once

#include <btas/type_traits.h>

namespace btas {

struct blas_lapack_impl_tag {};
struct generic_impl_tag     {};

template <typename... _Iterators>
struct blas_lapack_impl_delegator {
  using tag_type = 
    std::conditional_t< are_blas_lapack_compatible_v<_Iterators...>, 
                        blas_lapack_impl_tag,
                        generic_impl_tag >;
};


template <typename... _Iterators>
using blas_lapack_impl_t = 
  typename blas_lapack_impl_delegator<_Iterators...>::tag_type;

}
