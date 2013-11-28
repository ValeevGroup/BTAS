#ifndef __BTAS_AXPY_IMPL_H
#define __BTAS_AXPY_IMPL_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <btas/tensor_traits.h>
#include <btas/generic/types.h>

namespace btas {

/// Call BLAS depending on type of Tensor class
template<bool _DoAxpy> struct axpy_impl { };

/// Case that alpha is trivially multipliable to elements
template<> struct axpy_impl<true>
{
   typedef unsigned long size_type;

   template<typename _T, class _IteratorX, class _IteratorY>
   axpy_impl(const size_type& Nsize, const _T& alpha, _IteratorX itrX, _IteratorY itrY)
   {
      for (size_type i = 0; i < Nsize; ++i, ++itrX, ++itrY) (*itrY) += alpha * (*itrX);
   }
};

/// Case that alpha is multiplied recursively by AXPY
template<> struct axpy_impl<false>
{
   typedef unsigned long size_type;

   template<typename _T, class _IteratorX, class _IteratorY>
   axpy_impl(const size_type& Nsize, const _T& alpha, _IteratorX itrX, _IteratorY itrY)
   {
      for (size_type i = 0; i < Nsize; ++i, ++itrX, ++itrY) axpy(alpha, *itrX, *itrY);
   }
};

/// Generic implementation of BLAS-AXPY
/// tensor iterator must provide consecutive increment operator or never skip index
/// i.e. disable "std::set<T>::iterator" or something like that...
/// TODO: is there any missing type traits?
template<
   typename _T, class _TensorX, class _TensorY,
   class = typename std::enable_if<
      is_tensor<_TensorX>::value &
      is_tensor<_TensorY>::value &
      std::is_same<typename _TensorX::value_type,
                   typename _TensorY::value_type
      >::value
   >::type
>
void axpy(const _T& alpha, const _TensorX& x, _TensorY& y)
{
   if (x.empty())
   {
      y.clear();
      return;
   }

   if (y.empty())
   {
      y.resize(x.shape());
   }
   else
   {
      assert(std::equal(x.shape().begin(), x.shape().end(), y.shape().begin()));
   }

   typedef typename std::iterator_traits<typename _TensorX::iterator>::value_type value_type;
   axpy_impl<std::is_convertible<_T, value_type>::value> call(x.size(), alpha, x.begin(), y.begin());
}

} // namespace btas

#endif // __BTAS_AXPY_IMPL_H
