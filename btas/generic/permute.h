#ifndef __BTAS_PERMUTE_H
#define __BTAS_PERMUTE_H 1

#include <type_traits>

#include <btas/btas_types.h>
#include <btas/tarray.h>
#include <btas/reindex.h>

namespace btas {

template<typename _T, size_type _N>
TVector<_T, _N> permute (const TVector<_T, _N>& x, const Index<_N>& index)
{
   TVector<_T, _N> y;
   for (size_type i = 0; i < _N; ++i)
   {
      y[i] = x[index[i]];
   }
   return indexY;
}

template<typename _T, size_type _N>
void permute (const TArray<_T, _N>& x, const Index<_N>& index, TArray<_T, _N>& y)
{
   y.resize(permute(x.shape(), index));
   Reindex(x.data(), y.data(), permute(x.stride(), index), y.shape());
}

} // namespace btas

#endif // __BTAS_PERMUTE_H
