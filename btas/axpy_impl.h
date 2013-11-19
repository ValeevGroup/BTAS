#ifndef __BTAS_AXPY_IMPL_H
#define __BTAS_AXPY_IMPL_H 1

namespace btas {

/// Call BLAS depending on type of Tensor class
template<bool _DoAxpy> struct axpy_impl { };

/// Case that alpha is trivially multipliable to elements
template<> struct axpy_impl<true>
{
   template<typename _T, class _Tensor>
   axpy_impl(const _T& alpha, const _Tensor& x, _Tensor& y)
   {
      auto ix = x.begin();
      auto iy = y.begin();
      while(ix != x.end())
      {
         (*iy) += alpha * (*ix);
         ++ix;
         ++iy;
      }
   }
};

/// Case that alpha is multiplied recursively by AXPY
template<> struct axpy_impl<false>
{
   template<typename _T, class _Tensor>
   axpy_impl(const _T& alpha, const _Tensor& x, _Tensor& y)
   {
      auto ix = x.begin();
      auto iy = y.begin();
      while(ix != x.end())
      {
         axpy(alpha, *ix, *iy);
         ++ix;
         ++iy;
      }
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

   typedef typename std::iterator_traits<typename _Tensor::iterator>::value_type value_type;
// axpy_impl<std::is_same<_T, value_type>::value> call(alpha, x, y);
   axpy_impl<std::is_convertible<_T, value_type>::value> call(alpha, x, y);
}

} // namespace btas

#endif // __BTAS_AXPY_IMPL_H
