#ifndef __BTAS_REFERENCE_TENSOR_H
#define __BTAS_REFERENCE_TENSOR_H 1

#include <cassert>
#include <algorithm>
#include <type_traits>
#include <vector>

/// default storage type
template<typename _T>
using DEFAULT_STORAGE = std::vector<_T>;

/// default range type
using DEFAULT_RANGE   = std::vector<unsigned long>;

namespace btas {

/// reference implementation of dense tensor class (variable rank)
template<typename _T,
         class _Container = DEFAULT_STORAGE<_T>,
         class _Range = DEFAULT_RANGE>
class Tensor {

public:

   //  ========== Starting Public Interface and Its Reference Implementations ==========

   //
   //  Concepts for Standard Tensor Template Class
   //

   /// element type
   typedef _T value_type;

   /// type of array storing data as 1D array
   typedef _Container storage_type;

   /// size type
   typedef typename storage_type::size_type size_type;

   /// iterator
   typedef typename storage_type::iterator iterator;

   /// const iterator
   typedef typename storage_type::const_iterator const_iterator;

   /// type of array for index ranges
   /// range_type requires, default-constructible, resizable, accessible by operator[]
   typedef _Range range_type;

   /// type of array for index/shape
   /// shape_type requires, default-constructible, resizable, accessible by operator[]
   /// regarding shape_type = range_type, in this implementation
   typedef _Range shape_type;

public:

   //
   //  Constructors
   //

   /// default constructor
   /// \param n tensor rank
   explicit
   Tensor (size_type n = 0)
   : range_ (n, 0), stride_ (n, 0)
   { }

   /// destructor
   ~Tensor () { }

   /// constructor with index range
   template<typename _arg1, typename... _args>
   Tensor (const _arg1& first, const _args&... rest)
   {
      resize(first, rest...);
   }

   /// constructor with index range object
   Tensor (const range_type& range)
   {
      resize(range);
   }

   //
   //  Copy semantics:
   //  provides the interface b/w different tensor class
   //  _Tensor must have convertible value_type and iterators for range, stride, and data
   //

   /// copy constructor
   template<class _Tensor>
   Tensor (const _Tensor& x)
   : range_ (x.rank()), stride_ (x.rank())
   {
      std::copy(x.range().begin(), x.range().end(), range_.begin());

      std::copy(x.stride().begin(), x.stride().end(), stride_.begin());

      data_.resize(x.size());
      std::copy(x.begin(), x.end(), data_.begin());
   }

   /// copy assignment operator
   template<class _Tensor>
   Tensor&
   operator= (const _Tensor& x)
   {
      range_.resize(x.rank());
      std::copy(x.range().begin(), x.range().end(), range_.begin());

      stride_.resize(x.rank());
      std::copy(x.stride().begin(), x.stride().end(), stride_.begin());

      data_.resize(x.size());
      std::copy(x.begin(), x.end(), data_.begin());
      return *this;
   }

   /// move constructor
   Tensor (Tensor&& x)
   : range_ (x.range_), stride_ (x.stride_), data_ (x.data_)
   { }

   /// move assignment operator
   Tensor&
   operator= (Tensor&& x)
   {
      range_.swap(x.range_);
      stride_.swap(x.stride_);
      data_.swap(x.data_);
      return *this;
   }

   /// number of indices (tensor rank)
   size_type
   rank () const
   {
      return range_.size();
   }

   /// \return number of elements
   size_type
   size () const
   {
      return data_.size();
   }

   /// \return range object
   const range_type&
   range () const
   {
      return range_;
   }

   /// \return n-th range
   const typename range_type::value_type&
   range (const size_type& n) const
   {
      return range_[n];
   }

   /// \return shape (= range) object
   const shape_type&
   shape () const
   {
      return range_;
   }

   /// \return n-th shape (= range)
   const typename shape_type::value_type&
   shape (const size_type& n) const
   {
      return range_[n];
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

   /// \return element without range check
   template<typename... _args>
   const value_type& 
   operator() (const size_type& first, const _args&... rest) const
   {
      return data_[_address<0>(first, rest...)];
   }

   /// \return element without range check (rank() == general)
   const value_type& 
   operator() (const shape_type& index) const
   {
      return data_[_address(index)];
   }

   /// access element without range check
   template<typename... _args>
   value_type& 
   operator() (const size_type& first, const _args&... rest)
   {
      return data_[_address<0>(first, rest...)];
   }

   /// access element without range check (rank() == general)
   value_type& 
   operator() (const shape_type& index)
   {
      return data_[_address(index)];
   }
   
   /// \return element without range check
   template<typename... _args>
   const value_type& 
   at (const size_type& first, const _args&... rest) const
   {
      assert(_check_range<0>(first, rest...));
      return data_[_address<0>(first, rest...)];
   }

   /// \return element without range check (rank() == general)
   const value_type& 
   at (const shape_type& index) const
   {
      assert(_check_range(index));
      return data_[_address(index)];
   }

   /// access element without range check
   template<typename... _args>
   value_type& 
   at (const size_type& first, const _args&... rest)
   {
      assert(_check_range<0>(first, rest...));
      return data_[_address<0>(first, rest...)];
   }

   /// access element without range check (rank() == general)
   value_type& 
   at (const shape_type& index)
   {
      assert(_check_range(index));
      return data_[_address(index)];
   }
   
   /// resize array with range
   template<typename... _args>
   void
   resize (const typename range_type::value_type& first, const _args&... rest)
   {
      range_.resize(1u+sizeof...(rest));
      _set_range<0>(first, rest...);
      _set_stride();
      data_.resize(range_[0]*stride_[0]);
   }

   /// resize array with range object
   void
   resize (const range_type& range)
   {
      assert(range.size() > 0);
      range_ = range;
      _set_stride();
      data_.resize(range_[0]*stride_[0]);
   }

   /// swap this and x
   void 
   swap (Tensor& x)
   {
      range_.swap(x.range_);
      stride_.swap(x.stride_);
      data_.swap(x.data_);
   }

   /// clear all members
   void 
   clear()
   {
      range_.clear();
      stride_.clear();
      data_.clear();
   }

   //  ========== Finished Public Interface and Its Reference Implementations ==========

   //
   //  Here comes Non-Standard members
   //

   /// addition assignment
   Tensor&
   operator+= (const Tensor& x)
   {
      assert(std::equal(range_.begin(), range_.end(), x.range_.begin()));
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
      assert(std::equal(range_.begin(), range_.end(), x.range_.begin()));
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

   /// set range object
   template<size_type i, typename... _args>
   void
   _set_range (const typename range_type::value_type& first, const _args&... rest)
   {
      range_[i] = first;
      _set_range<i+1>(rest...);
   }

   /// set range object (specialized)
   template<size_type i>
   void
   _set_range (const typename range_type::value_type& first)
   {
      range_[i] = first;
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

   /// calculate stride_ from given range_
   void
   _set_stride ()
   {
      stride_.resize(range_.size());
      size_type str = 1;
      for(size_type i = range_.size()-1; i > 0; --i) {
         stride_[i] = str;
         str *= range_[i]; /* FIXME: this should be hacked for general purpose */
      }
      stride_[0] = str;
   }

   /// test whether index is in range
   template<size_type i, typename... _args>
   bool
   _check_range (const size_type& first, const _args&... rest) const
   {
      return (first >= 0 && first < range_[i] && _check_range<i+1>(rest...));
   }

   /// test whether index is in range
   template<size_type i>
   bool
   _check_range (const size_type& first) const
   {
      return (first >= 0 && first < range_[i]);
   }

   /// test whether index is in range
   bool
   _check_range (const shape_type& index)
   {
      assert(index.size() == rank());
      typename range_type::iterator r = range_.begin();
      return std::all_of(index.begin(), index.end(), [&r] (const typename shape_type::value_type& i) { return (i >= 0 && i < *r++); });
   }

private:

   //
   // Data members go here
   //

   range_type range_; ///< range (shape)

   shape_type stride_; ///< stride

   storage_type data_; ///< data stored as 1D array

};

}; // namespace btas

#endif // __BTAS_REFERENCE_TENSOR_H
