#ifndef __BTAS_ARRAYADAPTOR_H_
#define __BTAS_ARRAYADAPTOR_H_

// adaptors for std "array" containers

#include <vector>
#include <array>
#include <btas/numerictype.h>

namespace btas {

  /// Adaptor to sequence container, e.g. std::vector and btas::varray.
  template <typename Array>
  struct array_adaptor {
      typedef Array array;
      typedef typename Array::value_type value_type;

      static Array construct(std::size_t N) {
        return array(N);
      }
      static Array construct(std::size_t N,
                             value_type value) {
        return array(N, value);
      }
      static void resize(Array& x, std::size_t N) {
        x.resize(N);
      }
  };

  /// Adaptor to array
  template <typename T, size_t N>
  struct array_adaptor< std::array<T, N> > {
      typedef std::array<T, N> array;
      typedef typename array::value_type value_type;

      static std::array<T, N> construct(std::size_t n) {
        assert(n <= N);
        std::array<T, N> result;
        return result;
      }
      static std::array<T, N> construct(std::size_t n, T value) {
        assert(n <= N);
        std::array<T, N> result;
        std::fill_n(result.begin(), n, value);
        return result;
      }

      static void resize(array& x, std::size_t n) {
        assert(x.size() == N);
        assert(x.size() >= n);
      }

  };

}

#endif /* __BTAS_ARRAYADAPTOR_H_ */
