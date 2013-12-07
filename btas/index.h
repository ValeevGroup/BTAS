#ifndef __BTAS_INDEX_H_
#define __BTAS_INDEX_H_

#include <vector>
#include <array>

//
// Index types (aray, vector), and adaptors
//

namespace btas {

  template <typename Array>
  struct value_type;

  template <typename T>
  struct value_type< std::vector<T> > {
      typedef typename std::vector<T>::value_type value;
  };

  template <typename T, size_t N>
  struct value_type< std::array<T, N> > {
      typedef typename std::array<T, N>::value_type value;
  };

  /// Allows to construct std::array and std::vector with number of elements and value
  template <typename Array>
  struct construct {
      static Array call(std::size_t N, typename value_type<Array>::value value);
  };

  template <typename T>
  struct construct< std::vector<T> > {
      static std::vector<T> call(std::size_t N, T value = btas::NumericType<T>::zero()) {
        return std::vector<T>(N, value);
      }
  };

  template <typename T, size_t N>
  struct construct< std::array<T, N> > {
      static std::array<T, N> call(std::size_t n, T value = btas::NumericType<T>::zero()) {
        assert(n == N);
        std::array<T, N> result;
        std::fill(result.begin(), result.end(), value);
        return result;
      }
  };

}

#endif /* INDEX_H_ */
