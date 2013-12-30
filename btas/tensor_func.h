/*
 * tensor_func.h
 *
 *  Created on: Dec 30, 2013
 *      Author: evaleev
 */

#ifndef BTAS_TENSOR_FUNC_H_
#define BTAS_TENSOR_FUNC_H_

#include <btas/tensorview.h>

namespace btas {

  template<typename _T,
           class _Range,
           class _Storage,
           typename _Permutation
          >
  TensorView<_T, _Range, _Storage>
  permute( Tensor<_T, _Range, _Storage>& t,
           _Permutation p) {
      return TensorView<_T, _Range, _Storage>( permute(t.range(), p), t.storage() );
  }

  template<typename _T,
           class _Range,
           class _Storage,
           typename _Permutation
          >
  TensorView<_T, _Range, const _Storage>
  permute( const Tensor<_T, _Range, _Storage>& t,
           _Permutation p) {
      return TensorView<_T, _Range, const _Storage>( permute(t.range(), p), t.storage() );
  }

  template<typename _T,
           class _Range,
           class _Storage,
           typename _U
          >
  TensorView<_T, _Range, _Storage>
  permute( Tensor<_T, _Range, _Storage>& t,
           std::initializer_list<_U> p) {
      return TensorView<_T, _Range, _Storage>( permute(t.range(), p), t.storage() );
  }

  template<typename _T,
           class _Range,
           class _Storage,
           typename _U
          >
  TensorView<_T, _Range, _Storage>
  permute( const Tensor<_T, _Range, _Storage>& t,
           std::initializer_list<_U> p) {
      return TensorView<_T, _Range, const _Storage>( permute(t.range(), p), t.storage() );
  }

}


#endif /* BTAS_TENSOR_FUNC_H_ */
