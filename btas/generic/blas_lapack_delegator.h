#pragma once

#include <btas/type_traits.h>

namespace btas {

struct generic_impl_tag     {};


#ifdef BTAS_HAS_BLAS_LAPACK

struct blas_lapack_impl_tag {};

template <typename... _Iterators>
struct blas_lapack_impl_delegator {
  using tag_type = 
    std::conditional_t< are_blas_lapack_compatible_v<_Iterators...>, 
                        blas_lapack_impl_tag,
                        generic_impl_tag >;
};

#else

template <typename... _Iterators>
struct blas_lapack_impl_delegator {
  using tag_type = generic_impl_tag;
};

#endif


template <typename... _Iterators>
using blas_lapack_impl_t = 
  typename blas_lapack_impl_delegator<_Iterators...>::tag_type;

}
