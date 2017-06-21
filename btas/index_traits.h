#ifndef __BTAS_INDEX_TRAITS_H
#define __BTAS_INDEX_TRAITS_H 1

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

} // namespace btas

#endif // __BTAS_INDEX_TRAITS_H
