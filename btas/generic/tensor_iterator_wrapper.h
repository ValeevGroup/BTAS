#ifndef __BTAS_TENSOR_ITERATOR_WRAPPER_H
#define __BTAS_TENSOR_ITERATOR_WRAPPER_H 1

namespace btas {

template<bool _HasData>
struct tensor_iterator_wrapper
{
   template<class _Tensor>
   static auto begin(_Tensor& x) -> decltype(x.begin()) { return x.begin(); }

   template<class _Tensor>
   static auto end  (_Tensor& x) -> decltype(x.end  ()) { return x.end  (); }
};

template<>
struct tensor_iterator_wrapper<true>
{
   template<class _Tensor>
   static auto begin(_Tensor& x) -> decltype(x.data()) { return x.data(); }

   template<class _Tensor>
   static auto end  (_Tensor& x) -> decltype(x.data()) { return x.data()+x.size(); }
};

} // namespace btas

#endif // __BTAS_TENSOR_ITERATOR_WRAPPER_H
