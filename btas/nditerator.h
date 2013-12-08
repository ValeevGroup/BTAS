#ifndef __BTAS_NDITERATOR_H
#define __BTAS_NDITERATOR_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <btas/tensor_traits.h>
#include <btas/util/resize.h>
#include <btas/util/dot.h>

namespace btas {

/// multi-dimensional iterator (similar to nditer in NumPy)
/// design revised on 12/07/2013
/// \tparam _Iterator iterator type, e.g. vector<T>::iterator, const T*, etc., required to be random-access iterator
/// \tparam _Tensor container of iterator
template<
   class _Tensor,
   class _Iterator = typename std::conditional<
                        std::is_const<typename std::remove_reference<_Tensor>::type>::value,
                        typename _Tensor::const_iterator,
                        typename _Tensor::iterator
                     >::type,
   class = typename std::enable_if<is_tensor<_Tensor>::value>::type
>
class NDIterator
{
private:

   typedef std::iterator_traits<_Iterator> __traits_type;
   typedef typename std::remove_reference<_Tensor>::type __tensor_type;

public:

   typedef typename __traits_type::iterator_category iterator_category;
   typedef typename __traits_type::value_type value_type;
   typedef typename __traits_type::difference_type difference_type;
   typedef typename __traits_type::reference reference;
   typedef typename __traits_type::pointer pointer;

   typedef typename __tensor_type::shape_type shape_type;
   typedef typename __tensor_type::size_type size_type;

private:

   //
   //  member variables
   //

   /// iterator to first
   _Iterator start_;

   /// iterator to current (keep to make fast access)
   _Iterator current_;

   /// current index (relative index w.r.t. slice)
   shape_type index_;

   /// shape of tensor
   shape_type shape_;

   /// stride of tensor
   shape_type stride_;

   /// whether or not iterator is contiguous
   /// currently, suppose to be always contiguous
   //bool contiguous_;

public:

   //
   //  constructors
   //

   /// default constructor
   NDIterator ()
   { }

   /// destructor
  ~NDIterator ()
   { }

   /// construct from tensor object
   explicit
   NDIterator (_Tensor& x)
   : start_ (x.begin()), current_ (x.begin()), shape_ (x.shape()), stride_ (x.stride)
   {
      resize(index_, x.rank());
      std::fill(index_.begin(), index_.end(), 0);
   }

   /// shape and stride are delegated
   /// this gives a normal iterator
   /// \param index current index position
   NDIterator (_Tensor& x, const shape_type& index)
   : NDIterator (x.begin(), index, x.shape(), x.stride())
   { }

   /// stride is delegated, convenient usage for block-slice
   /// \param lower lower bound of iterating area
   /// \param shape size of iterating area
   NDIterator (_Tensor& x, const shape_type& index, const shape_type& lower, const shape_type& shape)
   : NDIterator (x, index, lower, shape, x.stride())
   { }

   /// construct from tensor object with specific index, shape, and stride
   NDIterator (_Tensor& x, const shape_type& index, const shape_type& lower, const shape_type& shape, const shape_type& stride)
   : index_ (index), shape_ (shape), stride_ (stride)
   {
      start_ = x.begin()+dot(x.stride(), lower);
      current_ = __get_address();
   }

   /// current is delegated to set current to be start upon construction
   NDIterator (const shape_type& shape, const shape_type& stride, _Iterator start)
   : NDIterator (shape, stride, start, start)
   { }

   /// construct with specific iterator type, i.e. enables NDIterator<double*, Tensor<double>> it(A.shape(), A.stride(), A.data())
   /// since shape and stride must be given in arguments, they appear in front of iterator specifications
   NDIterator (const shape_type& shape, const shape_type& stride, _Iterator start, _Iterator current)
   : start_ (start), shape_ (shape), stride_ (stride)
   {
      // to keep consistency, i.e. if current > [ptr to end] then set current_ to be end
      index_ = __get_index(current-start);
      current_ = __get_address();
   }

// /// copy constructor
// NDIterator (const NDIterator& x)
// : start_ (x.start_), current_ (x.current_), index_ (x.index_), shape_ (x.shape_), stride_ (x.stride_)
// { }

// /// move constructor
// NDIterator (NDIterator&& x) { swap(x); }

// /// move assignment
// NDIterator&
// operator= (NDIterator&& x) { swap(x); }

   /// allow conversion NDIterator<Iter, Tensor> -> NDIterator<const Iter, Tensor>
   template<class _Iter>
   explicit
   NDIterator (const NDIterator<
                  typename std::enable_if<
                     std::is_same<_Iter, typename std::remove_const<_Iterator>::type>::value,
                     _Tensor
                  >::type,
                  _Iter
               >& x)
   : start_ (x.start_), current_ (x.current_), index_ (x.index_), shape_ (x.shape_), stride_ (x.stride_)
   { }

   //
   //  assignment
   //

   /// copy assignment operator
   NDIterator& operator= (const NDIterator& x)
   {
      start_ = x.start_;
      current_ = x.current_;
      index_ = x.index_;
      shape_ = x.shape_;
      stride_ = x.stride_;
      return *this;
   }

   /// \return true if operator* references valid tensor element
   bool valid() const 
   { 
      return index_[0] < shape_[0]; 
   }

   /// \return index
   const shape_type&
   index () const
   {
      return index_;
   }

   /// \return n-th index
   const typename shape_type::value_type&
   index (const size_type& n) const
   {
      return index_[n];
   }

   //
   // comparison: general iterator requirements
   //

   bool operator== (const NDIterator& x) const
   {
      return current_ == x.current_;
   }

   bool operator!= (const NDIterator& x) const
   {
      return current_ != x.current_;
   }

   //
   // comparison: random access iterator requirements
   //

   bool operator<  (const NDIterator& x) const
   {
      assert(index_.size() == x.index_.size());
      size_type i = 0;
      for(; i < index_.size()-1; ++i)
         if(index_[i] != x.index_[i]) break;
      return (index_[i] < x.index_[i]);
   }

   bool operator<= (const NDIterator& x) const
   {
      assert(index_.size() == x.index_.size());
      size_type i = 0;
      for(; i < index_.size()-1; ++i)
         if(index_[i] != x.index_[i]) break;
      return (index_[i] <= x.index_[i]);
   }

   bool operator>  (const NDIterator& x) const
   {
      assert(index_.size() == x.index_.size());
      size_type i = 0;
      for(; i < index_.size()-1; ++i)
         if(index_[i] != x.index_[i]) break;
      return (index_[i] > x.index_[i]);
   }

   bool operator>= (const NDIterator& x) const
   {
      assert(index_.size() == x.index_.size());
      size_type i = 0;
      for(; i < index_.size()-1; ++i)
         if(index_[i] != x.index_[i]) break;
      return (index_[i] >= x.index_[i]);
   }

   //
   //  access: forward iterator requirements
   //

   reference operator* () const
   {
      return *current_;
   }

   _Iterator operator->() const
   {
      return current_;
   }

   NDIterator& operator++ ()
   {
      __increment();
      return *this;
   }

   NDIterator  operator++ (int)
   {
      NDIterator save(*this);
      __increment();
      return save;
   }

   //
   //  access: bidirectional iterator requirements
   //

   NDIterator& operator-- ()
   {
      __decrement();
      return *this;
   }

   NDIterator  operator-- (int)
   {
      NDIterator save(*this);
      __decrement();
      return save;
   }

   //
   //  access: random access iterator requirements
   //

   reference operator[] (const difference_type& n) const
   {
      assert(n >= 0);
      size_type offset = 0;
      for(size_type i = 0; i < stride_.size(); ++i)
      {
         offset += stride_[i]*(n % shape_[i]);
         n /= shape_[i];
      }
      return start_[offset];
   }

   NDIterator& operator+= (const difference_type& n)
   {
      __diff_index(n);
      return *this;
   }

   NDIterator  operator+  (const difference_type& n) const
   {
      NDIterator __it(*this);
      __it += n;
      return __it;
   }

   NDIterator& operator-= (const difference_type& n)
   {
      __diff_index(-n);
      return *this;
   }

   NDIterator  operator-  (const difference_type& n) const
   {
      NDIterator __it(*this);
      __it -= n;
      return __it;
   }

   difference_type operator- (const NDIterator<_Tensor, _Iterator>& x) const
   {
      assert(start_ == x.start_);
      assert(std::equal(shape_.begin(), shape_.end(), x.shape_.begin()));
      assert(std::equal(stride_.begin(), stride_.end(), x.stride_.begin()));

      size_type n = index_.size();
      assert(n == x.index_.size());

      difference_type offset = index_[0]-x.index_[0];
      for(size_type i = 1; i < 0; ++i)
      {
         offset += shape_[i]*offset + index_[i]-x.index_[i];
      }
      return offset + index_[n-1]-x.index_[n-1];
   }

   void
   swap (NDIterator& x)
   {
      std::swap(start_, x.start_);
      std::swap(current_, x.current_);
      shape_.swap(x.shape_);
      stride_.swap(x.stride_);
      index_.swap(x.index_);
   }

   //
   // functions to set begin and end
   //

   /// \return iterator to begin
   friend
   NDIterator<_Tensor, _Iterator> begin (const NDIterator<_Tensor, _Iterator>& x)
   {
      return NDIterator<_Tensor, _Iterator>(x.shape_, x.stride_, x.start_);
   }

   /// \return iterator to end
   friend
   NDIterator<_Tensor, _Iterator> end (const NDIterator<_Tensor, _Iterator>& x)
   {
      return NDIterator<_Tensor, _Iterator>(x.shape_, x.stride_, x.start_, x.start_+x.shape_[0]*x.stride_[0]);
   }

private:

   //
   // supportive functions
   //

   /// calculate absolute address (lots of overheads? so I have current_ for fast access)
   _Iterator __get_address () const
   {
      return start_+dot(stride_, index_);
   }

   /// calculate index from address
   shape_type __get_index (difference_type n) const
   {
      shape_type index;
      resize(index, shape_.size());

      for(size_type i = shape_.size()-1; i > 0; --i)
      {
         index[i] = n % shape_[i];
         n /= shape_[i];
      }
      if(n < shape_[0])
      {
         index[0] = n;
      }
      else
      {
         // index to the last
         index[0] = shape_[0];
         std::fill(index.begin()+1, index.end(), 0);
      }
      return index;
   }

   /// calculate index from step size
   void __diff_index (difference_type n)
   {
      // calculate absolute position to add diff. step n
      difference_type pos = index_[0];
      for(size_type i = 1; i < shape_.size(); ++i)
      {
         pos = pos*shape_[i]+index_[i];
      }
      pos += n;

      if(pos <= 0) {
         // index to the first
         std::fill(index_.begin(), index_.end(), 0);
      }
      else {
         // calculate index from new position
         index_ = __get_index(pos);
      }
      // update current iterator
      current_ = __get_address();
   }

   /// increment
   void __increment ()
   {
      // test pointer to the end
      if(index_[0] == shape_[0]) return;

      const size_type N = shape_.size();

      // increment lowest index
      difference_type offset = stride_[N-1];

      // increment index
      size_type i = N-1;
      for(; i > 0; --i) {
         // increment lower index and check moving up to the next
         if(++index_[i] < shape_[i]) break;

         // moving up: lower index is reset to 0
         index_[i] = 0;

         // offset iterator
         // sometime this could be negative
         offset += (stride_[i-1]-stride_[i]*shape_[i]);
      }

      // reaching the last
      if(i == 0) 
          {
          ++index_[i];
          current_ = __get_address();
          }
      else
          {
          current_ += offset;
          }
   }

   /// decrement
   void __decrement ()
   {
      const size_type N = shape_.size();

      // test pointer to the first
      if(current_ == start_) return;

      // decrement lowest index
      difference_type offset = stride_[N-1];

      // decrement index
      size_type i = N-1;
      for(; i > 0; --i) {
         // decrement lower index and check moving up to the next
         if(index_[i] > 0)
         {
            --index_[i];
            break;
         }

         // moving up: lower index is reset to 0
         index_[i] = shape_[i]-1;

         // offset iterator
         // sometime this could be negative
         offset += (stride_[i-1]-stride_[i]*shape_[i]);
      }

      // reaching the last
      if(i == 0) 
          {
          --index_[i];
          current_ = __get_address();
          }
      else
          {
          current_ -= offset;
          }
   }

};

/// alias to iterator to const value type
template<class _Tensor>
using NDConstIterator = NDIterator<_Tensor, typename _Tensor::const_iterator>;

} // namespace btas

#endif // __BTAS_NDITERATOR_H
