#ifndef __BTAS_INDEX_TRAITS_H
#define __BTAS_INDEX_TRAITS_H 1

#include <cstdint>
#include <iterator>
#include <type_traits>

#include <btas/type_traits.h>

namespace btas {

/// test T has integral value_type
template<class T>
class has_integral_value_type {
   /// true case
   template<class U, class = typename std::is_integral<typename U::value_type>::type >
   static std::true_type __test(typename U::value_type*);
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0))>::value;
};


/// test _Index conforms the TWG.Index concept
/// check only value_type and operator[]
template<typename ...>
class is_index;

template<typename _Index>
class is_index<_Index> {
public:
   static constexpr const bool
   value = has_integral_value_type<_Index>::value &
           is_container<_Index>::value;
};

template<typename _Index1, typename _Index2, typename ... Rest>
class is_index<_Index1,_Index2,Rest...> {
public:
   static constexpr const bool
   value = false;
};

template <size_t Width>
struct signed_int;
template <>
struct signed_int<8ul> {
  using type = int_fast8_t;
};
template <>
struct signed_int<16ul> {
  using type = int_fast16_t;
};
template <>
struct signed_int<32ul> {
  using type = int_fast32_t;
};
template <>
struct signed_int<64ul> {
  using type = int_fast64_t;
};
template <>
struct signed_int<128ul> {
  using type = int_fast64_t;
};
template <>
struct signed_int<256ul> {
  using type = int_fast64_t;
};
template <size_t Width> using signed_int_t = typename signed_int<Width>::type;

} // namespace btas

#endif // __BTAS_INDEX_TRAITS_H
