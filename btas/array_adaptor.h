#ifndef __BTAS_ARRAYADAPTOR_H_
#define __BTAS_ARRAYADAPTOR_H_

// adaptors for std "array" containers

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

#include <btas/features.h>

#ifdef BTAS_HAS_BOOST_CONTAINER
#include <boost/container/small_vector.hpp>
#endif

#include <btas/tensor_traits.h>
#include <btas/generic/numeric_type.h>
#include <btas/varray/varray.h>
#include <btas/serialization.h>

namespace btas {

  template <typename Array> struct array_adaptor;

  /// Adaptor from std::array
  template <typename T, size_t N>
  struct array_adaptor< std::array<T, N> > {
      typedef std::array<T, N> array;
      typedef typename array::value_type value_type;

      static array construct(std::size_t n) {
        assert(n <= N);
        array result;
        // fill elements n+1 ... N-1 with zeroes
        std::fill_n(result.begin() + n, N - n, value_type{});
        return result;
      }
      static array construct(std::size_t n, T value) {
        assert(n <= N);
        array result;
        std::fill_n(result.begin(), n, value);
        return result;
      }

      static void resize(array& x, std::size_t n) {
        assert(x.size() == N);
        assert(x.size() >= n);
      }

      static void print(const array& a, std::ostream& os) {
        os << "{";
        for(std::size_t i = 0; i != N; ++i) {
          os << a[i];
          if (i != (N - 1))
            os << ",";
        }
        os << "}";
      }
  };

  template <typename T, size_t N>
  constexpr std::size_t rank(const std::array<T, N>& x) noexcept {
    return N;
  }

  /// Adaptor from const-size array
  template <typename T, size_t N>
  struct array_adaptor< T[N] > {
      typedef T (array)[N];
      typedef T value_type;

      static void print(const array& a, std::ostream& os) {
        os << "{";
        for(std::size_t i = 0; i != N; ++i) {
          os << a[i];
          if (i != (N - 1))
            os << ",";
        }
        os << "}";
      }
  };

  template <typename T, size_t N>
  constexpr std::size_t rank(const T (&x)[N]) noexcept {
    return N;
  }

  template <typename T, size_t N>
  std::ostream& operator<<(std::ostream& os, const std::array<T, N>& x) {
    array_adaptor<std::array<T, N> >::print(x,os);
    return os;
  }

  /// Adaptors for sequence container, e.g. std::vector, btas::varray, and std::initializer_list

  template <typename Array, class = typename std::enable_if<not btas::is_tensor<Array>::value>::type>
  std::size_t rank(const Array& x) {
    return x.size();
  }

  template <typename Array>
  struct array_adaptor {
      typedef Array array;
      typedef typename Array::value_type value_type;

      static array construct(std::size_t N) {
        return array(N);
      }
      static array construct(std::size_t N,
                             value_type value) {
        array result(N);
        std::fill(result.begin(), result.end(), value);
        return result;
      }
      static void resize(array& x, std::size_t N) {
        x.resize(N);
      }
      static void print(const array& a, std::ostream& os) {
        std::size_t n = rank(a);
        os << "{";
        for(std::size_t i = 0; i != n; ++i) {
          os << a[i];
          if (i != (n - 1))
            os << ",";
        }
        os << "}";
      }
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const btas::varray<T>& x) {
    array_adaptor<btas::varray<T> >::print(x,os);
    return os;
  }

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const std::vector<T>& x) {
    array_adaptor<std::vector<T> >::print(x,os);
    return os;
  }

  template <typename T>
  std::ostream& operator<<(std::ostream& os, const std::initializer_list<T>& x) {
    array_adaptor<std::vector<T> >::print(x,os);
    return os;
  }

#ifdef BTAS_HAS_BOOST_CONTAINER
  template <typename T, size_t N>
  std::ostream& operator<<(std::ostream& os, const boost::container::small_vector<T,N>& x) {
    array_adaptor<boost::container::small_vector<T,N>>::print(x,os);
    return os;
  }
#endif

}

namespace std {

#if __cplusplus < 201402L // add C++14 components to make transition to C++14 easier
  template <typename T, size_t N>
  const T* cbegin(const T(&x)[N]) {
    return &x[0];
  }
  template <typename T, size_t N>
  const T* cend(const T(&x)[N]) {
    return &x[N];
  }
  template <typename T, size_t N>
  const T* rbegin(T(&x)[N]) {
    return &x[N-1];
  }
  template <typename T, size_t N>
  const T* rend(T(&x)[N]) {
    return &x[0] - 1;
  }

  template <typename T>
  const T* cbegin(const T* x) {
    return x;
  }
  template <typename T>
  const T* cbegin(T* x) {
    return x;
  }
  template <typename T>
  T* begin(T* x) {
    return x;
  }

  template <typename C>
  constexpr auto cbegin(const C& x) -> decltype(std::begin(x)) {
    return std::begin(x);
  }
  template <typename C>
  constexpr auto cend(const C& x) -> decltype(std::end(x)) {
    return std::end(x);
  }
  template <typename C>
  auto rbegin(C& x) -> decltype(x.rbegin()) {
    return x.rbegin();
  }
  template <typename C>
  auto rend(C& x) -> decltype(x.rend()) {
    return x.rend();
  }
#endif

#if __cplusplus <= 201402L // add useful bits to make transition to C++17 easier
  template <class C, typename std::enable_if<std::is_pointer<decltype(std::declval<C>().data())>::value>::type* = nullptr>
  constexpr auto data(C& c) -> decltype(c.data()) {
      return c.data();
  }
  template <class C, typename std::enable_if<std::is_pointer<decltype(std::declval<C>().data())>::value>::type* = nullptr>
  constexpr auto data(const C& c) -> decltype(c.data()) {
      return c.data();
  }
  template <class T, std::size_t N>
  constexpr T* data(T (&array)[N]) noexcept
  {
      return array;
  }
  template <class E>
  constexpr const E* data(const std::initializer_list<E>& il) noexcept
  {
      return il.begin();
  }

  template <class C, typename std::enable_if<std::is_integral<decltype(std::declval<C>().size())>::value>::type* = nullptr>
  constexpr auto size(const C& c) -> decltype(c.size())
  {
    return c.size();
  }
  template <class T, std::size_t N>
  constexpr std::size_t size(const T (&array)[N]) noexcept
  {
    return N;
  }
#endif

  template <typename T>
  struct make_unsigned<std::vector<T> > {
      typedef std::vector<typename make_unsigned<T>::type > type;
  };
  template <typename T>
  struct make_unsigned<std::initializer_list<T> > {
      typedef std::initializer_list<typename make_unsigned<T>::type > type;
  };
  template <typename T, size_t N>
  struct make_unsigned<std::array<T, N> > {
      typedef std::array<typename make_unsigned<T>::type, N> type;
  };
  template <typename T>
  struct make_unsigned<btas::varray<T> > {
      typedef btas::varray<typename make_unsigned<T>::type > type;
  };
#ifdef BTAS_HAS_BOOST_CONTAINER
  template <typename T, size_t N>
  struct make_unsigned<boost::container::small_vector<T,N> > {
      typedef boost::container::small_vector<typename make_unsigned<T>::type,N> type;
  };
#endif
  template <typename T, size_t N>
  struct make_unsigned<T[N]> {
      typedef typename make_unsigned<T>::type uT;
      typedef uT (type)[N];
  };

}

namespace btas {
  template <typename Array, typename T>
  struct replace_value_type;

  template <typename T, typename U>
  struct replace_value_type<std::vector<T>, U> {
      typedef std::vector<U> type;
  };
  template <typename T, typename U>
  struct replace_value_type<std::initializer_list<T>,U> {
      typedef std::initializer_list<U> type;
  };
  template <typename T, size_t N, typename U>
  struct replace_value_type<std::array<T, N>,U> {
      typedef std::array<U, N> type;
  };
  template <typename T, typename U>
  struct replace_value_type<btas::varray<T>,U> {
      typedef btas::varray<U> type;
  };
#ifdef BTAS_HAS_BOOST_CONTAINER
  template <typename T, size_t N, typename U>
  struct replace_value_type<boost::container::small_vector<T, N>,U> {
      typedef boost::container::small_vector<U, N> type;
  };
#endif
  template <typename T, size_t N, typename U>
  struct replace_value_type<T[N],U> {
      typedef U (type)[N];
  };
}

#ifdef BTAS_HAS_BOOST_SERIALIZATION
#ifndef BOOST_SERIALIZATION_STD_ARRAY // legacy switch to disable BTAS-provided serialization of std::array
#define BOOST_SERIALIZATION_STD_ARRAY
#  if BOOST_VERSION / 100 < 1056
namespace boost {
  namespace serialization {

    template<class Archive, class T, size_t N>
    void serialize(Archive & ar, std::array<T,N> & a, const unsigned int version)
    {
        ar & boost::serialization::make_array(a.data(), a.size());
    }

  } // namespace serialization
} // namespace boost
#  endif // boost < 1.56 does not serialize std::array ... provide our own
#endif // not defined BOOST_SERIALIZATION_STD_ARRAY? provide our own
#endif

#if defined(BTAS_HAS_BOOST_CONTAINER) && defined(BTAS_HAS_BOOST_SERIALIZATION)
namespace boost {
  namespace serialization {

  /// boost serialization for boost::container::small_vector
  template<class Archive, typename T, size_t N>
  void serialize (Archive& ar, boost::container::small_vector<T,N>& x, const unsigned int version)
  {
      boost::serialization::split_free(ar, x, version);
  }
  template<class Archive, typename T, size_t N>
  void save (Archive& ar, const boost::container::small_vector<T,N>& x, const unsigned int version)
  {
      const boost::serialization::collection_size_type count(x.size());
      ar << BOOST_SERIALIZATION_NVP(count);
      if (count != decltype(count)(0))
        ar << boost::serialization::make_array(x.data(), count);
  }
  template<class Archive, typename T, size_t N>
  void load (Archive& ar, boost::container::small_vector<T,N>& x, const unsigned int version)
  {
      boost::serialization::collection_size_type count;
      ar >> BOOST_SERIALIZATION_NVP(count);
      x.resize(count);
      if (count != decltype(count)(0))
        ar >> boost::serialization::make_array(x.data(), count);
  }

  } // namespace serialization
} // namespace boost
#endif

namespace madness {
  namespace archive {

    // Forward declarations
    template <class>
    class archive_array;
    template <class T>
    inline archive_array<T> wrap(const T*, unsigned int);

    template <class Archive, typename T, std::size_t N, typename A>
    struct ArchiveLoadImpl<Archive, boost::container::small_vector<T, N, A>> {
      static inline void load(const Archive& ar,
                              boost::container::small_vector<T, N, A>& x) {
        std::size_t n{};
        ar& n;
        x.resize(n);
        ar & madness::archive::wrap(x.data(),n);
      }
    };

    template <class Archive, typename T, std::size_t N, typename A>
    struct ArchiveStoreImpl<Archive, boost::container::small_vector<T, N, A>> {
      static inline void store(const Archive& ar,
                               const boost::container::small_vector<T, N, A>& x) {
        ar& x.size() & madness::archive::wrap(x.data(), x.size());
      }
    };

  }  // namespace archive
}  // namespace madness

#endif /* __BTAS_ARRAYADAPTOR_H_ */
