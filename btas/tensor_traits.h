#ifndef __BTAS_TENSOR_TRAITS_H
#define __BTAS_TENSOR_TRAITS_H 1

#include <iterator>
#include <type_traits>

namespace btas {

/// test T has data() member
template<class T>
class has_data {
   /// true case
   template<class U>
   static auto __test(U* p) -> decltype(p->data(), std::true_type());
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0))>::value;
};

/// test T has rank() member
template<class T>
class has_rank {
   /// true case
   template<class U>
   static auto __test(U* p) -> decltype(p->rank(), std::true_type());
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0))>::value;
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

/// test T has shape_type
template<class T>
class has_shape_type {
   /// true case
   template<class U>
   static std::true_type __test(typename U::shape_type*);
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0))>::value;
};

/// test T has container_type
template<class T>
class has_container_type {
   /// true case
   template<class U>
   static std::true_type __test(typename U::container_type*);
   /// false case
   template<class>
   static std::false_type __test(...);
public:
   static constexpr const bool value = std::is_same<std::true_type, decltype(__test<T>(0))>::value;
};

/// test _Tensor has a standard tensor concept
/// check only value_type, shape_type, container_type, and rank() member
template<class _Tensor>
class is_tensor {
public:
   static constexpr const bool
   value = has_value_type<_Tensor>::value & has_shape_type<_Tensor>::value &
           has_container_type<_Tensor>::value & has_rank<_Tensor>::value;
};

} // namespace btas

#endif // __BTAS_TENSOR_TRAITS_H
