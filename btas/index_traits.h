#ifndef __BTAS_INDEX_TRAITS_H
#define __BTAS_INDEX_TRAITS_H 1

#include <iterator>
#include <type_traits>

namespace btas {

/// test T has operator[] member
template<class T>
class has_squarebraket {
   /// true case
   template<class U>
   static auto __test(U* p, std::size_t i) -> decltype(p->operator[](i), std::true_type());
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0,std::size_t(0)))>::value;
};

/// test T has value_type
template<class T>
class has_value_type {
   /// true case
   template<class U>
   static std::true_type __test(typename U::value_type*);
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0))>::value;
};

/// test _Index conforms the TWG.Index concept
/// check only value_type and operator[]
template<class _Index>
class is_index {
public:
   static constexpr const bool
   value = has_value_type<_Index>::value & has_squarebraket<_Index>::value;
};

} // namespace btas

#endif // __BTAS_INDEX_TRAITS_H
