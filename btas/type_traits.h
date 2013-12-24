#ifndef __BTAS_TYPE_TRAITS_H
#define __BTAS_TYPE_TRAITS_H 1

#include <type_traits>

namespace btas {

  /// extends std::common_type to yield a signed integer type if one of the arguments is a signed type
  template <typename I0, typename I1>
  struct common_signed_type {
      typedef typename std::common_type<I0,I1>::type common_type;
      typedef typename std::conditional<
          std::is_signed<I0>::value || std::is_signed<I1>::value,
          typename std::make_signed<common_type>::type,
          common_type
        >::type type;
  }; // common_signed_type

} // namespace btas

#endif // __BTAS_TYPE_TRAITS_H
