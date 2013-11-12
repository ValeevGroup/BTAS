#ifndef __BTAS_SLICED_TENSOR_H
#define __BTAS_SLICED_TENSOR_H 1

#include <tensor.h>

using DEFAULT_INDEX = std::vector<unsigned long>;

namespace btas {

template<class _Tensor>
class Slice {

public:

   typedef typename _Tensor::value_type value_type;

   typedef _T* pointer;

   typedef _Index index_type;

   _sliced_array (pointer first, const index_type& lb, const index_type& ub)
   : start_ (first), lbound_ (lb), ubound_ (ub) { }

   pointer start_; ///< weak pointer to the first element of original

   index_type lbound_; ///< lower boundary of slice

   index_type ubound_; ///< upper boundary of slice

   struct sliced_iterator_base : public std::forward_iterator_tag {

      index_type current_;

      void operator++ ()
      {
         for (size_type i = this->rank()-1; i > 0; --i) {
            if (++current_[i] <= ubound_[i]) break;
            current_[i] = lbound_[i];
         }
      }
   }

   struct iterator : public std::forward_iterator_tag {
      value_type& operator* ()
      {
      }
   };

   struct const_iterator : public std::forward_iterator_tag {
      const value_type& operator* ()
      {
      }
   };
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

#endif // __BTAS_SLICED_TENSOR_H
