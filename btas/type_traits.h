#ifndef __BTAS_TYPE_TRAITS_H
#define __BTAS_TYPE_TRAITS_H 1

#include <type_traits>
#include <complex>

// C++20 extensions
// TODO: CPP guard if compiling with C++20
namespace std {
  template< class T >
  struct remove_cvref {
      typedef std::remove_cv_t<std::remove_reference_t<T>> type;
  };

  template< class T>
  using remove_cvref_t = typename remove_cvref<T>::type;
}

namespace btas {

  template <typename... Ts>
  struct make_void {
    using type = void;
  };
  template <typename... Ts>
  using void_t = typename make_void<Ts...>::type;

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

  /// test T has begin() member
  template<class T>
  class has_begin {
      /// true case
      template<class U>
      static auto __test(U* p) -> decltype(p->begin(), std::true_type());
      /// false case
      template<class >
      static std::false_type __test(...);
    public:
      static constexpr const bool value = std::is_same<std::true_type,
          decltype(__test<T>(0))>::value;
  };

  /// test T has end() member
  template<class T>
  class has_end {
      /// true case
      template<class U>
      static auto __test(U* p) -> decltype(p->end(), std::true_type());
      /// false case
      template<class >
      static std::false_type __test(...);
    public:
      static constexpr const bool value = std::is_same<std::true_type,
          decltype(__test<T>(0))>::value;
  };

  /// test T has value_type
  template<class T>
  class has_value_type {
      /// true case
      template<class U>
      static std::true_type __test(typename U::value_type*);
      /// false case
      template<class >
      static std::false_type __test(...);
    public:
      static constexpr const bool value = std::is_same<std::true_type,
          decltype(__test<T>(0))>::value;
  };

  /// test _C conforms to the standard Container concept; basic tests only
  template<class _C>
  class is_container {
    public:
      static constexpr const bool value = has_value_type<_C>::value
          & has_begin<_C>::value & has_end<_C>::value;
  };

  /// test T has operator[] member
  template<class T>
  class has_squarebraket {
      /// true case
      template<class U>
      static auto __test(
          U* p, std::size_t i) -> decltype(p->operator[](i), std::true_type());
      /// false case
      template<class >
      static std::false_type __test(...);
    public:
      static constexpr const bool value = std::is_same<std::true_type,
          decltype(__test<T>(0,std::size_t(0)))>::value;
  };







  // Checks if an iterator is random access
  template <typename _Iterator>
  struct is_random_access_iterator {
  private:
    using iterator_traits = std::iterator_traits<_Iterator>;
  public:
    static constexpr bool value = 
      std::is_same_v< typename iterator_traits::iterator_category, 
                      std::random_access_iterator_tag >;
  };
  
  template <typename _Iterator>
  inline constexpr bool is_random_access_iterator_v = 
    is_random_access_iterator<_Iterator>::value;


  // Checks whether a type is compatible with BLAS/LAPACK, i.e.
  // is is S (float) / D (double) / C (complex float) / Z (complex double)
  template<class T>
  struct is_blas_lapack_type { 
    static constexpr bool value = false;
  };

  template<>
  struct is_blas_lapack_type<float> {
    static constexpr bool value = true;
  };
  template<>
  struct is_blas_lapack_type<double> {
    static constexpr bool value = true;
  };
  template<>
  struct is_blas_lapack_type<std::complex<float>> {
    static constexpr bool value = true;
  };
  template<>
  struct is_blas_lapack_type<std::complex<double>> {
    static constexpr bool value = true;
  };

  template <typename T>
  inline constexpr bool is_blas_lapack_type_v =
    is_blas_lapack_type<T>::value;



  // Checks if an iterator decays to a BLAS/LAPACK compatible type
  // is_blas_lapack_type + is_random_access_iterator
  // TODO: Should be is_contiguous_iterator with C++20
  template <typename _Iterator>
  struct is_blas_lapack_compatible {
  private:
    using iterator_traits = std::iterator_traits<_Iterator>;
    using value_type = std::remove_cvref_t<typename iterator_traits::value_type>;
    static constexpr bool is_rai = is_random_access_iterator_v<_Iterator>;
    static constexpr bool is_blt = is_blas_lapack_type_v<value_type>; 
  public:
    static constexpr bool value = is_rai and is_blt;
  };
  
  template <typename _Iterator>
  inline constexpr bool is_blas_lapack_compatible_v =
    is_blas_lapack_compatible<_Iterator>::value;
  

  // Checks if a collection of iterators are all BLAS/LAPACK compatible
  template <typename... _Iterators>
  struct are_blas_lapack_compatible;
  
  template <typename _Iterator, typename... Tail>
  struct are_blas_lapack_compatible<_Iterator, Tail...> {
  private:
    static constexpr bool tail_is_compatible = 
      are_blas_lapack_compatible<Tail...>::value;
  public:
    static constexpr bool value = 
      is_blas_lapack_compatible_v<_Iterator> and tail_is_compatible;
  };
  
  
  template <typename _Iterator>
  struct are_blas_lapack_compatible<_Iterator> {
    static constexpr bool value = is_blas_lapack_compatible_v<_Iterator>;
  };
  
  template <typename... _Iterators>
  inline constexpr bool are_blas_lapack_compatible_v =
    are_blas_lapack_compatible<_Iterators...>::value;
  



  template <typename T>
  struct is_scalar_arithmetic {
    static constexpr bool value = std::is_arithmetic_v<T>;
  };

  template <typename T>
  struct is_scalar_arithmetic< std::complex<T> > {
    static constexpr bool value = std::is_arithmetic_v<T>;
  };

  template <typename T>
  inline constexpr bool is_scalar_arithmetic_v = is_scalar_arithmetic<T>::value;


  template <typename T>
  struct real_type {
    using type = T;
  };
  template <typename T>
  struct real_type<std::complex<T>> {
    using type = T;
  };

  template <typename T>
  using real_type_t = typename real_type<T>::type;

  // Convienience traits
  template <typename _Iterator>
  using iterator_difference_t = 
    typename std::iterator_traits<_Iterator>::difference_type;



  template <typename T>
  inline constexpr bool is_complex_type_v = not std::is_same_v< T, real_type_t<T> >;



  template <typename T, typename = std::void_t<>>
  struct has_numeric_type : public std::false_type { };
  template <typename T>
  struct has_numeric_type< T, std::void_t<typename T::numeric_type> > :
    public std::true_type { };

  template <typename T, typename = std::void_t<>>
  struct numeric_type;

  template <typename T>
  struct numeric_type<T, std::enable_if_t<!has_numeric_type<T>::value>> {
    using type = T;
  };
  template <typename T>
  struct numeric_type<T, std::enable_if_t<has_numeric_type<T>::value>> {
    using type = typename T::numeric_type;
  };





  } // namespace btas
  
  #endif // __BTAS_TYPE_TRAITS_H
