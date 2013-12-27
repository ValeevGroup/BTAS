#ifndef __BTAS_PERMUTE_H
#define __BTAS_PERMUTE_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <btas/types.h>
#include <btas/nditerator.h>
#include <btas/util/resize.h>

#include <btas/tensor.h>
#include <btas/tensor_traits.h>
#include <btas/index_traits.h>

namespace btas {

  /// permute \c X using permutation \c p, write result to \c Y
  template<class _Tensor, typename _Permutation,
           class = typename std::enable_if<is_boxtensor<_Tensor>::value && is_index<_Permutation>::value>::type>
  void permute(const _Tensor& X, const _Permutation& p, _Tensor& Y) {
    Y = _Tensor(permute(X.range(), p));
    auto itrX = X.begin();
    auto itrY = Y.begin();
    for (auto i : X.range()) {
      *(itrY + Y.range().ordinal(i)) = *itrX;
      ++itrX;
    }
  }

  /// permute \c X using permutation \c p, write result to \c Y
  template<class _Tensor, typename _T,
           class = typename std::enable_if<is_boxtensor<_Tensor>::value>::type>
  void permute(const _Tensor& X, std::initializer_list<_T> pi, _Tensor& Y) {
      btas::varray<_T> p(pi);
      permute(X,p,Y);
  }

/// indexed permutation
template<class _Tensor, class = typename std::enable_if<is_tensor<_Tensor>::value>::type>
void permute (const _Tensor& X, const typename _Tensor::shape_type& indexX,
                    _Tensor& Y, const typename _Tensor::shape_type& indexY)
{
   size_type Nrank = X.rank();

   // check rank
   assert(Nrank == indexX.size() && Nrank == indexY.size());

   // case: doesn't need to permute
   if (std::equal(indexX.begin(), indexX.end(), indexY.begin()))
   {
      Y = X; return;
   }

   // check index X
   typename _Tensor::shape_type __sort_indexX(indexX);
   std::sort(__sort_indexX.begin(), __sort_indexX.end());
   assert(std::unique(__sort_indexX.begin(), __sort_indexX.end()) == __sort_indexX.end());

   // check index Y
   typename _Tensor::shape_type __sort_indexY(indexY);
   std::sort(__sort_indexY.begin(), __sort_indexY.end());
   assert(std::unique(__sort_indexY.begin(), __sort_indexY.end()) == __sort_indexY.end());

   // check X & Y
   assert(std::equal(__sort_indexX.begin(), __sort_indexX.end(), __sort_indexY.begin()));

   // calculate permute index
   typename _Tensor::shape_type __permute_index;
   resize(__permute_index, Nrank);

   auto first = indexX.begin();
   auto last  = indexX.end();
   for(size_type i = 0; i < Nrank; ++i)
   {
      auto found = std::find(indexX.begin(), indexX.end(), indexY[i]);
      assert(found != last);
      __permute_index[i] = std::distance(first, found);
   }

   // call permute
   permute(X, __permute_index, Y);
}

} // namespace btas

#endif // __BTAS_PERMUTE_H
