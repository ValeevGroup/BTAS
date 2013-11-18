#ifndef __BTAS_BTAS_IMPL_H
#define __BTAS_BTAS_IMPL_H 1

namespace btas {

template<typename _T, class _Tensor>
void axpy(const _T& alpha, const _Tensor& x, _Tensor& y)
{
   axpy_impl<std::is_same<_T, typename _Tensor::value_type>::value> call;
}

} // namespace btas

#endif // __BTAS_BLAS_IMPL_H
