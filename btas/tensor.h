#ifndef __BTAS_REFERENCE_TENSOR_H
#define __BTAS_REFERENCE_TENSOR_H 1

#include <cassert>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

#include <tensor_traits.h>

/// default storage type
template<typename _T>
using DEFAULT_STORAGE = std::vector<_T>;

/// default shape type
using DEFAULT_SHAPE   = std::vector<unsigned long>;

namespace btas {

/// reference implementation of dense tensor class (variable rank)
template<typename _T,
         class _Container = DEFAULT_STORAGE<_T>,
         class _Shape = DEFAULT_SHAPE>
class Tensor {

public:

   //  ========== Starting Public Interface and Its Reference Implementations ==========

   //
   //  Concepts for Standard Tensor Template Class
   //

   /// element type
   typedef _T value_type;

   /// type of array storing data as 1D array
   typedef _Container container_type;

   /// size type
   typedef typename container_type::size_type size_type;

   /// iterator
   typedef typename container_type::iterator iterator;

   /// const iterator
   typedef typename container_type::const_iterator const_iterator;

   /// type of array for index shapes
   /// shape_type requires, default-constructible, resizable, accessible by operator[]
   typedef _Shape shape_type;

public:

   //
   //  Constructors
   //

   /// default constructor
   Tensor () { }

   /// destructor
   ~Tensor () { }

   /// constructor with index shape
   template<typename... _args>
   explicit
   Tensor (const size_type& first, const _args&... rest)
   {
      resize(first, rest...);
   }

   /// constructor with index shape object
   explicit
   Tensor (const shape_type& shape)
   {
      resize(shape);
   }

   //
   //  Copy semantics:
   //  provides the interface b/w different tensor class
   //  _Tensor must have convertible value_type and iterators for shape, stride, and data
   //

   /// copy constructor
   template<class _Tensor, class = typename std::enable_if<is_tensor<_Tensor>::value>::type>
   explicit
   Tensor (const _Tensor& x)
   : shape_ (x.rank()), stride_ (x.rank())
   {
      std::copy(x.shape().begin(), x.shape().end(), shape_.begin());

      std::copy(x.stride().begin(), x.stride().end(), stride_.begin());

      data_.resize(x.size());
      std::copy(x.begin(), x.end(), data_.begin());
   }

   /// copy constructor specialized (avoid implicit deletion of copy constructor)
   /// TODO: should be implemented in terms of efficient copy.
   explicit
   Tensor (const Tensor& x)
   : shape_ (x.rank()), stride_ (x.rank())
   {
      std::copy(x.shape().begin(), x.shape().end(), shape_.begin());

      std::copy(x.stride().begin(), x.stride().end(), stride_.begin());

      data_.resize(x.size());
      std::copy(x.begin(), x.end(), data_.begin());
   }

   /// copy assignment operator
   template<class _Tensor, class = typename std::enable_if<is_tensor<_Tensor>::value>::type>
   Tensor&
   operator= (const _Tensor& x)
   {
      shape_.resize(x.rank());
      std::copy(x.shape().begin(), x.shape().end(), shape_.begin());

      stride_.resize(x.rank());
      std::copy(x.stride().begin(), x.stride().end(), stride_.begin());

      data_.resize(x.size());
      std::copy(x.begin(), x.end(), data_.begin());
      return *this;
   }

   /// copy assignment operator (avoid implicit deletion of copy assignment)
   /// TODO: should be implemented in terms of efficient copy.
   Tensor&
   operator= (const Tensor& x)
   {
      shape_.resize(x.rank());
      std::copy(x.shape().begin(), x.shape().end(), shape_.begin());

      stride_.resize(x.rank());
      std::copy(x.stride().begin(), x.stride().end(), stride_.begin());

      data_.resize(x.size());
      std::copy(x.begin(), x.end(), data_.begin());
      return *this;
   }

   /// move constructor
   explicit
   Tensor (Tensor&& x)
   : shape_ (x.shape_), stride_ (x.stride_), data_ (x.data_)
   { }

   /// move assignment operator
   Tensor&
   operator= (Tensor&& x)
   {
      shape_.swap(x.shape_);
      stride_.swap(x.stride_);
      data_.swap(x.data_);
      return *this;
   }

   /// number of indices (tensor rank)
   size_type
   rank () const
   {
      return shape_.size();
   }

   /// \return number of elements
   size_type
   size () const
   {
      return data_.size();
   }

   /// \return shape object
   const shape_type&
   shape () const
   {
      return shape_;
   }

   /// \return n-th shape
   const typename shape_type::value_type&
   shape (const size_type& n) const
   {
      return shape_[n];
   }

   /// \return stride object
   const shape_type&
   stride () const
   {
      return stride_;
   }

   /// \return n-th stride
   const typename shape_type::value_type&
   stride (const size_type& n) const
   {
      return stride_[n];
   }

   /// test whether storage is empty
   bool
   empty() const
   {
      return data_.empty();
   }

   /// \return const iterator first
   const_iterator
   begin() const
   {
      return data_.begin();
   }

   /// \return const iterator last
   const_iterator
   end() const
   {
      return data_.end();
   }

   /// \return iterator first
   iterator
   begin()
   {
      return data_.begin();
   }

   /// \return iterator last
   iterator
   end()
   {
      return data_.end();
   }

   /// \return element without shape check
   template<typename... _args>
   const value_type& 
   operator() (const size_type& first, const _args&... rest) const
   {
      return data_[_address<0>(first, rest...)];
   }

   /// \return element without shape check (rank() == general)
   const value_type& 
   operator() (const shape_type& index) const
   {
      return data_[_address(index)];
   }

   /// access element without shape check
   template<typename... _args>
   value_type& 
   operator() (const size_type& first, const _args&... rest)
   {
      return data_[_address<0>(first, rest...)];
   }

   /// access element without shape check (rank() == general)
   value_type& 
   operator() (const shape_type& index)
   {
      return data_[_address(index)];
   }
   
   /// \return element without shape check
   template<typename... _args>
   const value_type& 
   at (const size_type& first, const _args&... rest) const
   {
      assert(_check_range<0>(first, rest...));
      return data_[_address<0>(first, rest...)];
   }

   /// \return element without shape check (rank() == general)
   const value_type& 
   at (const shape_type& index) const
   {
      assert(_check_range(index));
      return data_[_address(index)];
   }

   /// access element without shape check
   template<typename... _args>
   value_type& 
   at (const size_type& first, const _args&... rest)
   {
      assert(_check_range<0>(first, rest...));
      return data_[_address<0>(first, rest...)];
   }

   /// access element without shape check (rank() == general)
   value_type& 
   at (const shape_type& index)
   {
      assert(_check_range(index));
      return data_[_address(index)];
   }
   
   /// resize array with shape
   template<typename... _args>
   void
   resize (const size_type& first, const _args&... rest)
   {
      shape_.resize(1u+sizeof...(rest));
      _set_shape<0>(first, rest...);
      _set_stride();
      data_.resize(shape_[0]*stride_[0]);
   }

   /// resize array with shape object
   void
   resize (const shape_type& shape)
   {
      assert(shape.size() > 0);
      shape_ = shape;
      _set_stride();
      data_.resize(shape_[0]*stride_[0]);
   }

   /// swap this and x
   void 
   swap (Tensor& x)
   {
      shape_.swap(x.shape_);
      stride_.swap(x.stride_);
      data_.swap(x.data_);
   }

   /// clear all members
   void 
   clear()
   {
      shape_.clear();
      stride_.clear();
      data_.clear();
   }

   //  ========== Finished Public Interface and Its Reference Implementations ==========

   //
   //  Here comes Non-Standard members (to be discussed)
   //

   /// addition assignment
   Tensor&
   operator+= (const Tensor& x)
   {
      assert(std::equal(shape_.begin(), shape_.end(), x.shape_.begin()));
      std::transform(data_.begin(), data_.end(), x.data_.begin(), data_.begin(), std::plus<value_type>());
      return *this;
   }

   /// addition of tensors
   Tensor
   operator+ (const Tensor& x) const
   {
      Tensor y(*this); y += x;
      return y; /* automatically called move semantics */
   }

   /// subtraction assignment
   Tensor&
   operator-= (const Tensor& x)
   {
      assert(std::equal(shape_.begin(), shape_.end(), x.shape_.begin()));
      std::transform(data_.begin(), data_.end(), x.data_.begin(), data_.begin(), std::minus<value_type>());
      return *this;
   }

   /// subtraction of tensors
   Tensor
   operator- (const Tensor& x) const
   {
      Tensor y(*this); y -= x;
      return y; /* automatically called move semantics */
   }

   /// \return bare const pointer to the first element of data_
   /// this enables to call BLAS functions
   const value_type*
   data () const
   {
      return data_.data();
   }

   /// \return bare pointer to the first element of data_
   /// this enables to call BLAS functions
   value_type* 
   data()
   {
      return data_.data();
   }

   /// fill all elements by val
   void
   fill (const value_type& val)
   {
      std::fill(data_.begin(), data_.end(), val);
   }

   /// generate all elements by gen()
   template<class Generator>
   void 
   generate (Generator gen)
   {
      std::generate(data_.begin(), data_.end(), gen);
   }

private:

   //
   //  Supportive functions
   //

   /// set shape object
   template<size_type i, typename... _args>
   void
   _set_shape (const size_type& first, const _args&... rest)
   {
      shape_[i] = first;
      _set_shape<i+1>(rest...);
   }

   /// set shape object (specialized)
   template<size_type i>
   void
   _set_shape (const size_type& first)
   {
      shape_[i] = first;
   }
   
   /// \return address pointed by index
   template<size_type i, typename... _args>
   size_type
   _address (const size_type& first, const _args&... rest) const
   {
      return first*stride_[i] + _address<i+1>(rest...);
   }

   /// \return address pointed by index
   template<size_type i>
   size_type
   _address (const size_type& first) const
   {
      return first*stride_[i];
   }

   /// \return address pointed by index
   size_type
   _address (const shape_type& index) const
   {
      assert(index.size() == rank());
      size_type adr = 0;
      for(size_type i = 0; i < rank(); ++i) {
          adr += stride_[i]*index[i];
      }
      return adr;
   }

   /// calculate stride_ from given shape_
   void
   _set_stride ()
   {
      stride_.resize(shape_.size());
      size_type str = 1;
      for(size_type i = shape_.size()-1; i > 0; --i) {
         stride_[i] = str;
         str *= shape_[i];
      }
      stride_[0] = str;
   }

   /// test whether index is in range
   template<size_type i, typename... _args>
   bool
   _check_range (const size_type& first, const _args&... rest) const
   {
      return (first >= 0 && first < shape_[i] && _check_range<i+1>(rest...));
   }

   /// test whether index is in range
   template<size_type i>
   bool
   _check_range (const size_type& first) const
   {
      return (first >= 0 && first < shape_[i]);
   }

   /// test whether index is in range
   bool
   _check_range (const shape_type& index)
   {
      assert(index.size() == rank());
      typename shape_type::iterator r = shape_.begin();
      return std::all_of(index.begin(), index.end(), [&r] (const typename shape_type::value_type& i) { return (i >= 0 && i < *r++); });
   }

private:

   //
   // Data members go here
   //

   shape_type shape_; ///< shape

   shape_type stride_; ///< stride

   container_type data_; ///< data stored as 1D array

};

} // namespace btas

#endif // __BTAS_REFERENCE_TENSOR_H
