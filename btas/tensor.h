#ifndef __BTAS_TENSOR_H
#define __BTAS_TENSOR_H 1

#include <aliases.h>

namespace btas {

/// forward decl. SlicedTensor
template <typename T>
class SlicedTensor;

/// reference implementation of dense tensor class
template <typename T>
class Tensor {
public:

   /// element type
   typedef T value_type;

   /// type of array storing index ranges (sizes)
   typedef TensorRange range_type;

   /// type of array storing data as 1D array
   typedef VARIABLE_SIZE_ARRAY<T> storage_type;

   /// iterator
   typedef typename storage_type::iterator iterator;

   /// const iterator
   typedef typename storage_type::const_iterator const_iterator;

   /// size type
   typedef size_t size_type;

   /// default constructor
   Tensor () { }

   /// constructor with index range, for rank() == 1
   explicit 
   Tensor (size_type n01)
   {
      resize (n01);
   }

   /// constructor with index range, for rank() == 2
   Tensor (size_type n01, size_type n02)
   {
      resize (n01, n02);
   }

   /// constructor with index range, for rank() == 3
   Tensor (size_type n01, size_type n02, size_type n03)
   {
      resize (n01, n02, n03);
   }

   /// constructor with index range, for rank() == 4
   Tensor (size_type n01, size_type n02, size_type n03, size_type n04)
   {
      resize (n01, n02, n03, n04);
   }

   /// constructor with index range, for rank() == 5
   Tensor (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05)
   {
      resize (n01, n02, n03, n04, n05);
   }

   /// constructor with index range, for rank() == 6
   Tensor (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06)
   {
      resize (n01, n02, n03, n04, n05, n06);
   }

   /// constructor with index range, for rank() == 7
   Tensor (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07)
   {
      resize (n01, n02, n03, n04, n05, n06, n07);
   }

   /// constructor with index range, for rank() == 8
   Tensor (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08)
   {
      resize (n01, n02, n03, n04, n05, n06, n07, n08);
   }

   /// constructor with index range, for rank() == 9
   Tensor (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08,
           size_type n09)
   {
      resize (n01, n02, n03, n04, n05, n06, n07, n08, n09);
   }

   /// constructor with index range, for rank() == 10
   Tensor (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08,
           size_type n09, size_type n10)
   {
      resize (n01, n02, n03, n04, n05, n06, n07, n08, n09, n10);
   }

   /// constructor with index range, for rank() == 11
   Tensor (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08,
           size_type n09, size_type n10, size_type n11)
   {
      resize (n01, n02, n03, n04, n05, n06, n07, n08, n09, n10, n11);
   }

   /// constructor with index range, for rank() == 12
   Tensor (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08,
           size_type n09, size_type n10, size_type n11, size_type n12)
   {
      resize (n01, n02, n03, n04, n05, n06, n07, n08, n09, n10, n11, n12);
   }

   /// copy constructor
   Tensor (const Tensor& x)
   : range_ (x.range_), stride_ (x.stride_)
   {
      data_.resize(x.size());
      NUMERIC_TYPE<T>::copy (x.size(), x.data_.data(), data_.data());
   }

   /// copy assignment operator
   Tensor&
   operator= (const Tensor& x)
   {
      range_ = x.range_;
      stride_ = x.stride_;
      data_.resize(x.size());
      NUMERIC_TYPE<T>::copy (x.size(), x.data_.data(), data_.data());
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
   rank () const { return range_.size(); }

   /// list of index sizes/ranges
   range_type
   range () const { return range_; }

   /// specific index size/range
   size_type
   range (size_type i) const { return range_.at(i); }

   /// list of strides
   range_type
   stride () const { return stride_; }

   /// specific stride
   size_type
   stride (size_type i) const { return stride_; }

   /// empty() -> storage_type::empty()
   bool
   empty() const { return data_.empty(); }

   /// addition assignment
   Tensor&
   operator+= (const Tensor& x)
   {
      assert (range_ == x.range_);
      NUMERIC_TYPE<T>::plus (x.size(), x.data_.data(), this->data_.data());
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
      assert (range_ == x.range_);
      NUMERIC_TYPE<T>::minus(x.size(), x.data_.data(), this->data_.data());
      return *this;
   }

   /// subtraction of tensors
   Tensor
   operator- (const Tensor& x) const
   {
      Tensor y(*this); y -= x;
      return y; /* automatically called move semantics */
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

   /// \return element i01 without range check (rank() == 1)
   const value_type& 
   operator() (size_type i01) const
   {
      range_type index = { i01 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 2)
   const value_type& 
   operator() (size_type i01, size_type i02) const
   {
      range_type index = { i01, i02 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 3)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03) const
   {
      range_type index = { i01, i02, i03 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 4)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04) const
   {
      range_type index = { i01, i02, i03, i04 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 5)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 6)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 7)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 8)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 9)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08,
               size_type i09) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 10)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08,
               size_type i09, size_type i10) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 11)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08,
               size_type i09, size_type i10, size_type i11) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10, i11 };
      return this->operator()(index);
   }

   /// \return element i01,i02,... without range check (rank() == 12)
   const value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08,
               size_type i09, size_type i10, size_type i11, size_type i12) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10, i11, i12 };
      return this->operator()(index);
   }

   /// \return element without range check (rank() == general)
   const value_type& 
   operator()(const range_type& index) const
   {
      assert(index.size() == this->rank());
      return data_[dot(index, stride_)];
   }
   
   /// access element i01 without range check (rank() == 1)
   value_type& 
   operator() (size_type i01)
   {
      range_type index = { i01 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 2)
   value_type& 
   operator() (size_type i01, size_type i02)
   {
      range_type index = { i01, i02 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 3)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03)
   {
      range_type index = { i01, i02, i03 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 4)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04)
   {
      range_type index = { i01, i02, i03, i04 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 5)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05)
   {
      range_type index = { i01, i02, i03, i04,
                           i05 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 6)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 7)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 8)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 9)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08,
               size_type i09)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 10)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08,
               size_type i09, size_type i10)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 11)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08,
               size_type i09, size_type i10, size_type i11)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10, i11 };
      return this->operator()(index);
   }

   /// access element i01,i02,... without range check (rank() == 12)
   value_type& 
   operator() (size_type i01, size_type i02, size_type i03, size_type i04,
               size_type i05, size_type i06, size_type i07, size_type i08,
               size_type i09, size_type i10, size_type i11, size_type i12)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10, i11, i12 };
      return this->operator()(index);
   }

   /// access element without range check (rank() == general)
   value_type& 
   operator() (const range_type& index)
   {
      assert(index.size() == this->rank());
      return data_[dot(index, stride_)];
   }
   
   /// \return element i01 with range check (rank() == 1)
   const value_type& 
   at (size_type i01) const
   {
      range_type index = { i01 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 2)
   const value_type& 
   at (size_type i01, size_type i02) const
   {
      range_type index = { i01, i02 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 3)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03) const
   {
      range_type index = { i01, i02, i03 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 4)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04) const
   {
      range_type index = { i01, i02, i03, i04 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 5)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 6)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 7)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 8)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 9)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08,
       size_type i09) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 10)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08,
       size_type i09, size_type i10) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 11)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08,
       size_type i09, size_type i10, size_type i11) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10, i11 };
      return at(index);
   }

   /// \return element i01,i02,... with range check (rank() == 12)
   const value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08,
       size_type i09, size_type i10, size_type i11, size_type i12) const
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10, i11, i12 };
      return at(index);
   }

   /// \return element with range check (rank() == general)
   const value_type& 
   at(const range_type& index) const
   {
      assert(index.size() == this->rank());
      return data_.at(dot(index, stride_));
   }
    
   /// access element i01 with range check (rank() == 1)
   value_type& 
   at (size_type i01)
   {
      range_type index = { i01 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 2)
   value_type& 
   at (size_type i01, size_type i02)
   {
      range_type index = { i01, i02 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 3)
   value_type& 
   at (size_type i01, size_type i02, size_type i03)
   {
      range_type index = { i01, i02, i03 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 4)
   value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04)
   {
      range_type index = { i01, i02, i03, i04 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 5)
   value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05)
   {
      range_type index = { i01, i02, i03, i04,
                           i05 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 6)
   value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 7)
   value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 8)
   value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 9)
   value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08,
       size_type i09)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 10)
   value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08,
       size_type i09, size_type i10)
   {
       range_type index = { i01, i02, i03, i04,
                            i05, i06, i07, i08,
                            i09, i10 };
       return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 11)
   value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08,
       size_type i09, size_type i10, size_type i11)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10, i11 };
      return at(index);
   }

   /// access element i01,i02,... with range check (rank() == 12)
   value_type& 
   at (size_type i01, size_type i02, size_type i03, size_type i04,
       size_type i05, size_type i06, size_type i07, size_type i08,
       size_type i09, size_type i10, size_type i11, size_type i12)
   {
      range_type index = { i01, i02, i03, i04,
                           i05, i06, i07, i08,
                           i09, i10, i11, i12 };
      return at(index);
   }

   /// access element with range check (rank() == general)
   value_type& 
   at(const range_type& index)
   {
      assert(index.size() == this->rank());
      return data_.at(dot(index, stride_));
   }
   
   /// \return number of elements
   size_type
   size() const
   {
      return data_.size();
   }

   /// \returns const posize_typeer to start of elements
   const value_type* 
   data() const
   {
      return data_.data();
   }

   /// \returns posize_typeer to start of elements
   value_type* 
   data()
   {
      return data_.data();
   }

   /// resize array range, for rank() == 1
   void 
   resize (size_type n01)
   {
      range_type r = { n01 };
      resize(r);
   }

   /// resize array range, for rank() == 2
   void 
   resize (size_type n01, size_type n02)
   {
      range_type r = { n01, n02 };
      resize(r);
   }

   /// resize array range, for rank() == 3
   void 
   resize (size_type n01, size_type n02, size_type n03)
   {
      range_type r = { n01, n02, n03 };
      resize(r);
   }

   /// resize array range, for rank() == 4
   void 
   resize (size_type n01, size_type n02, size_type n03, size_type n04)
   {
      range_type r = { n01, n02, n03, n04 };
      resize(r);
   }

   /// resize array range, for rank() == 5
   void 
   resize (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05)
   {
      range_type r = { n01, n02, n03, n04,
                       n05 };
      resize(r);
   }

   /// resize array range, for rank() == 6
   void 
   resize (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06)
   {
      range_type r = { n01, n02, n03, n04,
                       n05, n06 };
      resize(r);
   }

   /// resize array range, for rank() == 7
   void 
   resize (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07)
   {
      range_type r = { n01, n02, n03, n04,
                       n05, n06, n07 };
      resize(r);
   }

   /// resize array range, for rank() == 8
   void 
   resize (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08)
   {
      range_type r = { n01, n02, n03, n04,
                       n05, n06, n07, n08 };
      resize(r);
   }

   /// resize array range, for rank() == 9
   void 
   resize (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08,
           size_type n09)
   {
      range_type r = { n01, n02, n03, n04,
                       n05, n06, n07, n08,
                       n09 };
      resize(r);
   }

   /// resize array range, for rank() == 10
   void 
   resize (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08,
           size_type n09, size_type n10)
   {
      range_type r = { n01, n02, n03, n04,
                       n05, n06, n07, n08,
                       n09, n10 };
      resize(r);
   }

   /// resize array range, for rank() == 11
   void 
   resize (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08,
           size_type n09, size_type n10, size_type n11)
   {
      range_type r = { n01, n02, n03, n04,
                       n05, n06, n07, n08,
                       n09, n10, n11 };
      resize(r);
   }

   /// resize array range, for rank() == 12
   void 
   resize (size_type n01, size_type n02, size_type n03, size_type n04,
           size_type n05, size_type n06, size_type n07, size_type n08,
           size_type n09, size_type n10, size_type n11, size_type n12)
   {
      range_type r = { n01, n02, n03, n04,
                       n05, n06, n07, n08,
                       n09, n10, n11, n12 };
      resize(r);
   }

   /// resize array range, for general rank
   void
   resize (const range_type& range)
   {
      assert(range.size() > 0);
      range_ = range;
      size_type str = 1;
      for(size_type i = range_.size()-1; i > 0; --i) {
          stride_[i] = str;
          str *= range_[i];
      }
      stride_[0] = str;
      data_.resize(range_[0]*str);
   }

   /// \return sliced tensor
   /// TODO: have not yet been implemented
   SlicedTensor<value_type> 
   slice (const range_type& lbound, const range_type& ubound) const;

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

private:

   /// enable boost serialization
   friend class boost::serialization::access;

   /// serialize members
   template<class Archive>
   void serialize(Archive& ar, const unsigned int version)
   {
      ar & range_ & stride_ & data_;
   }

protected:

   // data members go here

   range_type range_; ///< range (shape)

   range_type stride_; ///< stride

   storage_type data_; ///< data stored as 1D array

};

}; //namespace btas

#ifndef __BTAS_SLICED_TENSOR_H

#include <sliced_tensor.h>

#endif

#endif // __BTAS_TENSOR_H
