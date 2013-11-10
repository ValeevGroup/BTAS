#ifndef __BTAS_VARRAY_H
#define __BTAS_VARRAY_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <boost/serialization/serialization.hpp>

namespace btas {

/// iterator class designed for varray
/// refferred from __gnu_cxx::__normal_iterator implementation

using std::iterator_traits;
using std::iterator;
template <typename _Iterator, typename _Container>
class __normal_iterator {
protected:
   _Iterator _M_current;

   typedef iterator_traits<_Iterator> __traits_type;

public:
   typedef _Iterator iterator_type;
   typedef typename __traits_type::iterator_category iterator_category;
   typedef typename __traits_type::value_type value_type;
   typedef typename __traits_type::difference_type difference_type;
   typedef typename __traits_type::reference reference;
   typedef typename __traits_type::pointer pointer;

   __normal_iterator() : _M_current(_Iterator()) { }

   explicit
   __normal_iterator(const _Iterator& __i) : _M_current(__i) { }

   // Allow iterator to const_iterator conversion
   template <typename _Iter>
   __normal_iterator(const __normal_iterator<_Iter,
      typename std::enable_if<(std::is_same<_Iter,
      typename _Container::pointer>::value), _Container>::type>& __i)
   : _M_current(__i.base()) { }

   // Forward iterator requirements
   reference
   operator*() const
   { return *_M_current; }

   pointer
   operator->() const
   { return _M_current; }

   __normal_iterator&
   operator++()
   {
      ++_M_current;
      return *this;
   }

   __normal_iterator
   operator++(int)
   { return __normal_iterator(_M_current++); }

   // Bidirectional iterator requirements
   __normal_iterator&
   operator--()
   {
      --_M_current;
      return *this;
   }

   __normal_iterator
   operator--(int)
   { return __normal_iterator(_M_current--); }

   // Random access iterator requirements
   reference
   operator[](const difference_type& __n) const
   { return _M_current[__n]; }

   __normal_iterator&
   operator+=(const difference_type& __n)
   { _M_current += __n; return *this; }

   __normal_iterator
   operator+(const difference_type& __n) const
   { return __normal_iterator(_M_current + __n); }

   __normal_iterator&
   operator-=(const difference_type& __n)
   { _M_current -= __n; return *this; }

   __normal_iterator
   operator-(const difference_type& __n) const
   { return __normal_iterator(_M_current - __n); }

   const _Iterator&
   base() const
   { return _M_current; }
};

// Forward iterator requirements
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator==(const __normal_iterator<_IteratorL, _Container>& __lhs, const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() == __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator==(const __normal_iterator<_Iterator, _Container>& __lhs, const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() == __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator!=(const __normal_iterator<_IteratorL, _Container>& __lhs, const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() != __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator!=(const __normal_iterator<_Iterator, _Container>& __lhs, const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() != __rhs.base(); }

// Random access iterator requirements
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator<(const __normal_iterator<_IteratorL, _Container>& __lhs, const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() < __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator<(const __normal_iterator<_Iterator, _Container>& __lhs, const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() < __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator>(const __normal_iterator<_IteratorL, _Container>& __lhs, const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() > __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator>(const __normal_iterator<_Iterator, _Container>& __lhs, const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() > __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator<=(const __normal_iterator<_IteratorL, _Container>& __lhs, const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() <= __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator<=(const __normal_iterator<_Iterator, _Container>& __lhs, const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() <= __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator>=(const __normal_iterator<_IteratorL, _Container>& __lhs, const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() >= __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator>=(const __normal_iterator<_Iterator, _Container>& __lhs, const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() >= __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline typename __normal_iterator<_IteratorL, _Container>::difference_type
operator-(const __normal_iterator<_IteratorL, _Container>& __lhs, const __normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() - __rhs.base(); }

template<typename _Iterator, typename _Container>
inline typename __normal_iterator<_Iterator, _Container>::difference_type
operator-(const __normal_iterator<_Iterator, _Container>& __lhs, const __normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() - __rhs.base(); }

template<typename _Iterator, typename _Container>
inline __normal_iterator<_Iterator, _Container>
operator+(typename __normal_iterator<_Iterator, _Container>::difference_type __n, const __normal_iterator<_Iterator, _Container>& __i)
{ return __normal_iterator<_Iterator, _Container>(__i.base() + __n); }

}; // namespace btas



namespace btas {

/// variable size array class without capacity info
/// NOTE: to reduce object size, this doesn't have the virtual destructor
template <typename T>
class varray {
public:

   typedef T value_type;
   typedef value_type& reference;
   typedef const value_type& const_reference;
   typedef value_type* pointer;
   typedef const value_type* const_pointer;
   typedef __normal_iterator<pointer, varray> iterator;
   typedef __normal_iterator<const_pointer, varray> const_iterator;
   typedef std::reverse_iterator<iterator> reverse_iterator;
   typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
   typedef typename std::iterator_traits<iterator>::difference_type difference_type;
   typedef size_t size_type;

private:

   friend class boost::serialization::access;
   /// boost serialization: load as varray
   template<class Archive>
   void load (Archive& ar, varray& x, const unsigned int version)
   {
      size_type n; ar >> n;
      x.resize(n);
      for (value_type& xi : x) ar >> xi;
   }

   /// boost serialization: save as varray
   template<class Archive>
   void save (Archive& ar, const varray& x, const unsigned int version)
   {
      ar << x.size();
      for (const value_type& xi : x) ar << xi;
   }

protected:

   pointer _M_start;

   pointer _M_finish;

public:

   varray ()
   : _M_start (nullptr), _M_finish (nullptr)
   { }

  ~varray ()
   {
      if (!empty()) delete [] _M_start;
   }

   explicit
   varray (size_type n)
   : _M_start (nullptr), _M_finish (nullptr)
   {
      if (n > 0) {
         _M_start = new value_type [n];
         _M_finish = _M_start + n;
      }
   }

   varray (size_type n, const value_type& val)
   : _M_start (nullptr), _M_finish (nullptr)
   {
      if (n > 0) {
         _M_start = new value_type [n];
         _M_finish = _M_start + n;
         std::fill (_M_start, _M_finish, val);
      }
   }

   template <class InputIterator>
   varray (InputIterator first, InputIterator last)
   : _M_start (nullptr), _M_finish (nullptr)
   {
      size_type n = static_cast<size_type>(last - first);
      if (n > 0) {
         _M_start = new value_type [n];
         _M_finish = _M_start + n;
         std::copy (_M_start, first, last);
      }
   }

   varray (const varray& x)
   : _M_start (nullptr), _M_finish (nullptr)
   {
      size_type n = x.size();
      if (n > 0) {
         _M_start = new value_type [n];
         _M_finish = _M_start + n;
         std::copy (_M_start, x._M_start, x._M_finish);
      }
   }

   varray (varray&& x)
   : _M_start (x._M_start), _M_finish (x._M_finish)
   {
      x._M_start = nullptr;
      x._M_finish = nullptr;
   }

   varray (std::initializer_list<value_type> il)
   : _M_start (nullptr), _M_finish (nullptr)
   {
      size_type n = il.size();
      if (n > 0) {
         _M_start = new value_type [n];
         _M_finish = _M_start + n;
         std::copy (_M_start, il.begin(), il.end());
      }
   }

   varray& operator= (const varray& x) {
      if (!empty()) {
         delete [] _M_start;
         _M_start = nullptr;
         _M_finish = nullptr;
      }

      size_type n = x.size();
      if (n > 0) {
         _M_start = new value_type [n];
         _M_finish = _M_start + n;
         std::copy (_M_start, x._M_start, x._M_finish);
      }
      return *this;
   }

   varray& operator= (varray&& x)
   {
      swap (x); // if something in this object, it will be destructed by x
      return *this;
   }

   varray& operator= (std::initializer_list<value_type> il)
   {
      if (!empty()) {
         delete [] _M_start;
         _M_start = nullptr;
         _M_finish = nullptr;
      }

      size_type n = il.size();
      if (n > 0) {
         _M_start = new value_type [n];
         _M_finish = _M_start + n;
         std::copy (_M_start, il.begin(), il.end());
      }
   }

   iterator begin () noexcept
   {
      return iterator (_M_start);
   }

   const_iterator begin () const noexcept
   {
      return const_iterator (_M_start);
   }

   iterator end () noexcept
   {
      return iterator (_M_finish);
   }

   const_iterator end () const noexcept
   {
      return const_iterator (_M_finish);
   }

   reverse_iterator rbegin () noexcept
   {
      return reverse_iterator (end());
   }

   const_reverse_iterator rbegin () const noexcept
   {
      return const_reverse_iterator (end());
   }

   reverse_iterator rend () noexcept
   {
      return reverse_iterator (begin());
   }

   const_reverse_iterator rend () const noexcept
   {
      return const_reverse_iterator (begin());
   }

   size_type size () const noexcept
   { return _M_finish - _M_start; }

   void resize (size_type n)
   {
      if (!empty()) {
         delete [] _M_start;
         _M_start = nullptr;
         _M_finish = nullptr;
      }

      if (n > 0) {
         _M_start = new value_type [n];
         _M_finish = _M_start + n;
      }
   }

   void resize (size_type n, const value_type& val)
   {
      if (!empty()) {
         delete [] _M_start;
         _M_start = nullptr;
         _M_finish = nullptr;
      }

      if (n > 0) {
         _M_start = new value_type [n];
         _M_finish = _M_start + n;
         std::fill (_M_start, _M_finish, val);
      }
   }

   bool empty () const noexcept
   { return (size() == 0); }

   reference operator [] (size_type n)
   { return *(_M_start + n); }

   const_reference operator [] (size_type n) const
   { return *(_M_start + n); }

   reference at (size_type n)
   {
      pointer _M_current = _M_start + n;
      assert (_M_current < _M_finish);
      return *_M_current;
   }

   const_reference at (size_type n) const
   {
      pointer _M_current = _M_start + n;
      assert (_M_current < _M_finish);
      return *_M_current;
   }

   reference front ()
   { return *_M_start; }

   const_reference front () const
   { return *_M_start; }

   reference back ()
   { return *(_M_finish - 1); }

   const_reference back () const
   { return *(_M_finish - 1); }

   value_type* data () noexcept
   {
      return _M_start;
   }

   const value_type* data () const noexcept
   {
      return _M_start;
   }

   void swap (varray& x)
   {
      std::swap (_M_start, x._M_start);
      std::swap (_M_finish, x._M_finish);
   }

   void clear ()
   {
      if (!empty()) {
         delete [] _M_start;
         _M_start = nullptr;
         _M_finish = nullptr;
      }
   }
};

};

#endif // __BTAS_VARRAY_H
