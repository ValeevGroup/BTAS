#ifndef __BTAS_ALIASES_H
#define __BTAS_ALIASES_H

#ifdef _USE_STL_VECTOR

#include <vector>
#include <boost/serialization/vector.hpp>
namespace btas { template <typename T> using VARIABLE_SIZE_ARRAY = std::vector<T>; };

#else

#include <varray.h>
namespace btas { template <typename T> using VARIABLE_SIZE_ARRAY = varray<T>; };

#endif

#include <numeric_type.h>

#include <cassert>
#define BTAS_ASSERT(truth) assert(truth)
#define BTAS_STATIC_ASSERT(truth) static_assert(truth)

namespace btas {

/// dot product of arrays
template <typename T>
T dot (const VARIABLE_SIZE_ARRAY<T>& x, const VARIABLE_SIZE_ARRAY<T>& y)
{
   assert (x.size() == y.size());

   auto xi = x.begin();
   auto yi = y.begin();

   T xy = NUMERIC_TYPE<T>::zero();

   for (; xi != x.end(); ++xi, ++yi)
      xy += (*xi) * (*yi);

   return xy;
}

/// unsigned integer type
typedef unsigned long UINT;

/// range object type
typedef VARIABLE_SIZE_ARRAY<UINT> TensorRange;

}; // namespace btas

#endif // __BTAS_ALIASES_H
