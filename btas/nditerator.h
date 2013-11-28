#ifndef __BTAS_NDITERATOR_H
#define __BTAS_NDITERATOR_H 1

#include <iterator>
#include <type_traits>
//#include <algorithm>

#include <btas/tensor_traits.h>
#include <btas/resize.h>

namespace btas {

template<typename _Iterator, class _Shape, bool _IsResizable = is_resizable<_Shape>::value> class NDIterator { };

/// multi-dimensional iterator (similar to nditer in NumPy???) with variable-size shape object
/// provides iterator over tensor elements with specific shape & stride
/// which enables to doing permutation, reshape, tie, slicing, etc...
/// to enable NDIteration, _Iterator must be a random-access iterator
template<typename _Iterator, class _Shape>
class NDIterator<_Iterator, _Shape, true>
{

private:

   typedef std::iterator_traits<_Iterator> __traits_type;

public:

   typedef typename __traits_type::iterator_category iterator_category;
   typedef typename __traits_type::value_type value_type;
   typedef typename __traits_type::difference_type difference_type;
   typedef typename __traits_type::reference reference;
   typedef typename __traits_type::pointer pointer;

   typedef _Shape shape_type;

   typedef unsigned long size_type;

private:

   //
   //  member variables
   //

   /// iterator to first
   _Iterator start_;

   /// iterator to current (keep to make fast access)
   _Iterator current_;

   /// shape of tensor
   shape_type shape_;

   /// stride of tensor
   shape_type stride_;

   /// current index (relative index w.r.t. slice)
   shape_type index_;

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

   /// construct with the least arguments
   NDIterator (_Iterator start, const shape_type& shape)
   : start_ (start), current_ (start), shape_ (shape)
   {
      index_.resize(shape_.size());
      std::fill(index_.begin(), index_.end(), 0);
      __set__stride ();
   }

   /// construct with stride hack
   NDIterator (_Iterator start, const shape_type& shape, const shape_type& stride)
   : start_ (start), current_ (start), shape_ (shape), stride_ (stride)
   {
      index_.resize(shape_.size());
      std::fill(index_.begin(), index_.end(), 0);
   }

   /// construct with specific index position
   NDIterator (_Iterator start, const shape_type& shape, const shape_type& stride, const shape_type& index)
   : start_ (start), shape_ (shape), stride_ (stride), index_ (index)
   {
      assert(index_[0] <= shape_[0]);
      for(size_type i = 1; i < shape_.size(); ++i)
      {
         assert(index_[i] < shape_[i]);
      }
      current_ = __get__address();
   }

   /// copy constructor
   NDIterator (const NDIterator& x)
   : start_ (x.start_), current_ (x.current_), shape_ (x.shape_), stride_ (x.stride_), index_ (x.index_)
   { }

   //
   //  assignment
   //

   /// copy assignment operator
   NDIterator& operator= (const NDIterator& x)
   {
      start_ = x.start_;
      current_ = x.current_;
      shape_ = x.shape_;
      stride_ = x.stride_;
      index_ = x.index_;
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
      __diff__index(n);
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
      __diff__index(-n);
      return *this;
   }

   NDIterator  operator-  (const difference_type& n) const
   {
      NDIterator __it(*this);
      __it -= n;
      return __it;
   }

private:

   //
   // supportive functions
   //

   /// calculate stride from shape
   void __set__stride ()
   {
      stride_.resize(shape_.size());
      size_type str = 1;
      for(size_type i = shape_.size()-1; i > 0; --i)
      {
         stride_[i] = str;
         str *= shape_[i];
      }
      stride_[0] = str;
   }

   /// calculate absolute address (lots of overheads? so I have current_ for fast access)
   _Iterator __get__address () const
   {
      difference_type offset = 0;
      for(size_type i = 0; i < stride_.size(); ++i)
      {
         offset += stride_[i]*index_[i];
      }
      return start_+offset;
   }

   /// calculate index from step size
   void __diff__index (difference_type n)
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
         for(size_type i = shape_.size()-1; i > 0; --i)
         {
            index_[i] = pos % shape_[i];
            pos /= shape_[i];
         }
         if(pos < shape_[0])
         {
            index_[0] = pos;
         }
         else
         {
            // index to the last
            index_[0] = shape_[0];
            std::fill(index_.begin()+1, index_.end(), 0);
         }
      }
      // update current iterator
      current_ = __get__address();
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
         current_ = __get__address();
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
         ++index_[i];
         current_ = __get__address();
      }
      else
      {
         current_ -= offset;
      }
   }

};

} // namespace btas

#endif // __BTAS_NDITERATOR_H
