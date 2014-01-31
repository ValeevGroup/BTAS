/*
 * tensor_func.h
 *
 *  Created on: Dec 30, 2013
 *      Author: evaleev
 */

#ifndef BTAS_TENSOR_FUNC_H_
#define BTAS_TENSOR_FUNC_H_


namespace btas {

  // Maps Tensor     -> TensorView,
  //      TensorView -> TensorView
  // appropriately transferring constness of the storage, that is,
  // if _T is const, uses const _T::storage_type, otherwise just _T::storage_type
  template<typename _T>
  using TensorViewOf = TensorView<typename _T::value_type,
                                  typename _T::range_type,
                                  typename std::conditional<std::is_const<_T>::value,
                                                            const typename _T::storage_type,
                                                            typename _T::storage_type
                                                           >::type>;

  template<typename _T,
           typename _Permutation>
  TensorViewOf<_T>
  permute( _T& t,
           _Permutation p) {
      return TensorViewOf<_T>( permute(t.range(), p), t.storage() );
  }

  template<typename _T,
           typename _U>
  TensorViewOf<_T>
  permute( _T& t,
           std::initializer_list<_U> p) {
      return TensorViewOf<_T>( permute(t.range(), p), t.storage() );
  }

  template <typename _T>
  TensorViewOf<_T>
  diag(_T& T)
    {
    return TensorViewOf<_T>(diag(T.range()),T.storage());
    }

  template <typename _T>
  TensorViewOf<_T>
  diag(_T&& T)
    {
    return TensorViewOf<_T>(diag(T.range()),T.storage());
    }

  template <typename _T>
  TensorViewOf<_T>
  group(_T& T,
        size_t istart,
        size_t iend)
    {
    return TensorViewOf<_T>(group(T.range(),istart,iend),T.storage());
    }

  template <typename _T>
  TensorViewOf<_T>
  group(_T&& T,
        size_t istart,
        size_t iend)
    {
    return TensorViewOf<_T>(group(T.range(),istart,iend),T.storage());
    }

  template <typename _T>
  TensorViewOf<_T>
  flatten(_T& T)
    {
    return TensorViewOf<_T>(flatten(T.range()),T.storage());
    }

  template <typename _T>
  TensorViewOf<_T>
  flatten(_T&& T)
    {
    return TensorViewOf<_T>(flatten(T.range()),T.storage());
    }

} // namespace btas


#endif /* BTAS_TENSOR_FUNC_H_ */
