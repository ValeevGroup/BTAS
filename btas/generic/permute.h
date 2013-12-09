#ifndef __BTAS_PERMUTE_H
#define __BTAS_PERMUTE_H 1

#include <type_traits>

#include <btas/types.h>
#include <btas/nditerator.h>
#include <btas/util/resize.h>

namespace btas {

template<class _Tensor, class = typename std::enable_if<is_tensor<_Tensor>::value>::type>
void permute (const _Tensor& X, const typename _Tensor::shape_type& index, _Tensor& Y)
{
   typedef typename _Tensor::shape_type shape_type;

   shape_type shapeY; resize(shapeY, index.size());
   shape_type strX2Y; resize(strX2Y, index.size());

   for (size_type i = 0; i < index.size(); ++i)
   {
      shapeY[i] = X.shape (index[i]);
      strX2Y[i] = X.stride(index[i]);
   }

   Y.resize(shapeY);

   NDIterator<_Tensor, typename _Tensor::const_iterator> itrX(shapeY, strX2Y, X.begin());

   for (auto itrY = Y.begin(); itrY != Y.end(); ++itrX, ++itrY)
   {
      *itrY = *itrX;
   }
}

} // namespace btas

#endif // __BTAS_PERMUTE_H
