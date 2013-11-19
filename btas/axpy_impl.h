#ifndef __BTAS_AXPY_IMPL_H
#define __BTAS_AXPY_IMPL_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

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
template<typename _T, class _Tensor>
void axpy(const _T& alpha, const _Tensor& x, _Tensor& y)
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

   static_assert(std::is_same<typename std::iterator_traits<typename _Tensor::iterator>::iterator_category, std::random_access_iterator_tag>::value, "axpy: _Tensor::iterator must be random-access iterator");

   typedef typename std::iterator_traits<typename _Tensor::iterator>::value_type value_type;
   axpy_impl<std::is_convertible<_T, value_type>::value> call(x.size(), alpha, x.begin(), y.begin());
}

} // namespace btas

#endif // __BTAS_AXPY_IMPL_H
