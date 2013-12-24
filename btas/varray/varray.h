#ifndef __BTAS_VARRAY_H
#define __BTAS_VARRAY_H 1

#include <algorithm>

#include <boost/serialization/serialization.hpp>

#include <btas/varray/memory_reference.h>

namespace btas {

/// variable size array class without capacity info
/// NOTE: to reduce object size, this doesn't have the virtual destructor
/// TODO: add allocator to template parameter
template <typename T>
class varray {
public:

   typedef memory_reference<T> container;

   typedef typename container::value_type value_type;
   typedef typename container::reference reference;
   typedef typename container::const_reference const_reference;
   typedef typename container::pointer pointer;
   typedef typename container::const_pointer const_pointer;
   typedef typename container::iterator iterator;
   typedef typename container::const_iterator const_iterator;
   typedef typename container::reverse_iterator reverse_iterator;
   typedef typename container::const_reverse_iterator const_reverse_iterator;
   typedef typename container::difference_type difference_type;
   typedef typename container::size_type size_type;
   typedef varray<T> eval_type;

private:

   container data_;

   friend class boost::serialization::access;

public:

   varray ()
   : data_ (nullptr, nullptr)
   { }

   ~varray ()
   {
      if (!data_.empty()) delete [] data_._M_start;
   }

   explicit
   varray (size_type n)
   : data_ (nullptr, nullptr)
   {
      if (n > 0) {
         data_._M_start = new value_type [n];
         data_._M_finish = data_._M_start + n;
      }
   }

   varray (size_type n, const value_type& val)
   : data_ (nullptr, nullptr)
   {
      if (n > 0) {
         data_._M_start = new value_type [n];
         data_._M_finish = data_._M_start + n;
         std::fill (data_._M_start, data_._M_finish, val);
      }
   }

   template <class InputIterator>
   varray (InputIterator first, InputIterator last)
   : data_ (nullptr, nullptr)
   {
      size_type n = static_cast<size_type>(last - first);
      if (n > 0) {
         data_._M_start = new value_type [n];
         data_._M_finish = data_._M_start + n;
         std::copy (first, last, data_._M_start);
      }
   }

   varray (const varray& x)
   : data_ (nullptr, nullptr)
   {
      size_type n = x.size();
      if (n > 0) {
         data_._M_start = new value_type [n];
         data_._M_finish = data_._M_start + n;
         std::copy (x.data_._M_start, x.data_._M_finish, data_._M_start);
      }
   }

   varray (varray&& x)
   : data_ (x.data_)
   {
      x.data_._M_start = nullptr;
      x.data_._M_finish = nullptr;
   }

   template <typename U, class = typename std::enable_if< std::is_convertible<U, value_type>::value >::type >
   varray (std::initializer_list<U> il)
   : data_ (nullptr, nullptr)
   {
      size_type n = il.size();
      if (n > 0) {
         data_._M_start = new value_type [n];
         data_._M_finish = data_._M_start + n;
         std::copy (il.begin(), il.end(), data_._M_start);
      }
   }

   varray& operator= (const varray& x) {
      if (!empty()) {
         delete [] data_._M_start;
         data_._M_start = nullptr;
         data_._M_finish = nullptr;
      }

      size_type n = x.size();
      if (n > 0) {
         data_._M_start = new value_type [n];
         data_._M_finish = data_._M_start + n;
         std::copy (x.data_._M_start, x.data_._M_finish, data_._M_start);
      }
      return *this;
   }

   varray& operator= (varray&& x)
   {
      swap (x); // if something in this object, it will be destructed by x
      return *this;
   }

   template <typename U, class = typename std::enable_if< std::is_convertible<U, value_type>::value >::type >
   varray& operator= (std::initializer_list<U> il)
   {
      if (!empty()) {
         delete [] data_._M_start;
         data_._M_start = nullptr;
         data_._M_finish = nullptr;
      }

      size_type n = il.size();
      if (n > 0) {
         data_._M_start = new value_type [n];
         data_._M_finish = data_._M_start + n;
         std::copy (il.begin(), il.end(), data_._M_start);
      }
   }

   iterator begin () noexcept
   {
      return data_.begin();
   }

   const_iterator begin () const noexcept
   {
      return data_.begin();
   }

   const_iterator cbegin () const noexcept
   {
      return data_.cbegin();
   }

   iterator end () noexcept
   {
      return data_.end();
   }

   const_iterator end () const noexcept
   {
      return data_.end();
   }

   const_iterator cend () const noexcept
   {
      return data_.cend();
   }

   reverse_iterator rbegin () noexcept
   {
      return data_.rbegin();
   }

   const_reverse_iterator rbegin () const noexcept
   {
      return data_.rbegin();
   }

   reverse_iterator rend () noexcept
   {
      return data_.rend();
   }

   const_reverse_iterator rend () const noexcept
   {
      return data_.rend();
   }

   size_type size () const noexcept
   { return data_.size(); }

   void resize (size_type n)
   {
      if (!empty()) {
         delete [] data_._M_start;
         data_._M_start = nullptr;
         data_._M_finish = nullptr;
      }

      if (n > 0) {
         data_._M_start = new value_type [n];
         data_._M_finish = data_._M_start + n;
      }
   }

   void resize (size_type n, const value_type& val)
   {
      if (!empty()) {
         delete [] data_._M_start;
         data_._M_start = nullptr;
         data_._M_finish = nullptr;
      }

      if (n > 0) {
         data_._M_start = new value_type [n];
         data_._M_finish = data_._M_start + n;
         std::fill (data_._M_start, data_._M_finish, val);
      }
   }

   bool empty () const noexcept
   { return data_.empty(); }

   reference operator [] (size_type n)
   { return data_[n]; }

   const_reference operator [] (size_type n) const
   { return data_[n]; }

   reference at (size_type n)
   { return data_.at(n); }

   const_reference at (size_type n) const
   { return data_.at(n); }

   reference front ()
   { return data_.front(); }

   const_reference front () const
   { return data_.front(); }

   reference back ()
   { return data_.back(); }

   const_reference back () const
   { return data_.back(); }

   value_type* data () noexcept
   { return data_.data(); }

   const value_type* data () const noexcept
   { return data_.data(); }

   void swap (varray& x)
   { std::swap (data_, x.data_); }

   void clear ()
   {
      if (!empty()) {
         delete [] data_._M_start;
         data_._M_start = nullptr;
         data_._M_finish = nullptr;
      }
   }
};

template <typename T>
inline bool operator== (const btas::varray<T>& a,
                        const btas::varray<T>& b) {
  return std::equal(a.begin(), a.end(), b.begin());
}

template <typename T>
inline bool operator!= (const btas::varray<T>& a,
                        const btas::varray<T>& b) {
  return not (a == b);
}

namespace boost {
namespace serialization {

  /// boost serialization: load as varray
  template<class Archive, typename T>
  void load (Archive& ar, btas::varray<T>& x, const unsigned int version)
  {
      typename btas::varray<T>::size_type n; ar >> n;
      x.resize(n);
      for (typename btas::varray<T>::value_type& xi : x) ar >> xi;
  }

  /// boost serialization: save as varray
  template<class Archive, typename T>
  void save (Archive& ar, const btas::varray<T>& x, const unsigned int version)
  {
      ar << x.size();
      for (const typename btas::varray<T>::value_type& xi : x) ar << xi;
  }
}
}

};

template <typename T>
inline bool operator== (const btas::varray<T>& a,
                        const btas::varray<T>& b) {
  return std::equal(a.begin(), a.end(), b.begin());
}

template <typename T>
inline bool operator!= (const btas::varray<T>& a,
                        const btas::varray<T>& b) {
  return not (a == b);
}

#endif // __BTAS_VARRAY_H
