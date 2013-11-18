#ifndef __BTAS_SLICE_H
#define __BTAS_SLICE_H 1

#include <tensor.h>

using DEFAULT_INDEX = std::vector<unsigned long>;

namespace btas {

template<class _Tensor, class _Range = btas::Default::range_type>
class Slice {

public:

   typedef typename _Tensor::value_type value_type;

   typedef typename _Tensor::shape_type shape_type;

   typedef _Range range_type;

   typedef range_iterator<typename _Tensor::iterator> iterator;

   typedef range_iterator<typename _Tensor::const_iterator> const_iterator;

   // something like?
   iterator begin() { return iterator(ptr_->begin(), ptr_->shape(), range_); }

private:

   _Tensor* ptr_; ///< weak pointer to the first element of original

   range_type range_; ///< contains ranges of slice

};

/// expression template of sliced tensor object
template <class _Tensor>
class SlicedTensor {

private:

public:

   typedef _Tesnor tensor_type;

   typedef _Tensor* pointer_type;

   typedef typename _Tensor::range_type range_type;

   typedef typename _Tensor::shape_type shape_type;

private:

   pointer_type ptensor_; ///< pointer to original tensor

   shape_type lbound_; ///< lower boundary to be sliced

   shape_type ubound_; ///< upper boundary to be sliced

};

}; //namespace btas

#endif // __BTAS_SLICE_H
