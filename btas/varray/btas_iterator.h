#ifndef __BTAS_ITERATOR_H
#define __BTAS_ITERATOR_H 1

#include <iterator>
#include <type_traits>

namespace btas {

/// iterator skeleton designed for various iterators in BTAS
/// refferred from __gnu_cxx::__normal_iterator implementation

using std::iterator_traits;
using std::iterator;
template <typename _Iterator, typename _Container>
class __btas_normal_iterator {
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

   __btas_normal_iterator() : _M_current(_Iterator()) { }

   explicit
   __btas_normal_iterator(const _Iterator& __i) : _M_current(__i) { }

   // Allow iterator to const_iterator conversion
   template <typename _Iter>
   __btas_normal_iterator(const __btas_normal_iterator<_Iter,
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

   __btas_normal_iterator&
   operator++()
   {
      ++_M_current;
      return *this;
   }

   __btas_normal_iterator
   operator++(int)
   { return __btas_normal_iterator(_M_current++); }

   // Bidirectional iterator requirements
   __btas_normal_iterator&
   operator--()
   {
      --_M_current;
      return *this;
   }

   __btas_normal_iterator
   operator--(int)
   { return __btas_normal_iterator(_M_current--); }

   // Random access iterator requirements
   reference
   operator[](const difference_type& __n) const
   { return _M_current[__n]; }

   __btas_normal_iterator&
   operator+=(const difference_type& __n)
   { _M_current += __n; return *this; }

   __btas_normal_iterator
   operator+(const difference_type& __n) const
   { return __btas_normal_iterator(_M_current + __n); }

   __btas_normal_iterator&
   operator-=(const difference_type& __n)
   { _M_current -= __n; return *this; }

   __btas_normal_iterator
   operator-(const difference_type& __n) const
   { return __btas_normal_iterator(_M_current - __n); }

   const _Iterator&
   base() const
   { return _M_current; }
};

// Forward iterator requirements
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator==(const __btas_normal_iterator<_IteratorL, _Container>& __lhs, const __btas_normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() == __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator==(const __btas_normal_iterator<_Iterator, _Container>& __lhs, const __btas_normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() == __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator!=(const __btas_normal_iterator<_IteratorL, _Container>& __lhs, const __btas_normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() != __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator!=(const __btas_normal_iterator<_Iterator, _Container>& __lhs, const __btas_normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() != __rhs.base(); }

// Random access iterator requirements
template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator<(const __btas_normal_iterator<_IteratorL, _Container>& __lhs, const __btas_normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() < __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator<(const __btas_normal_iterator<_Iterator, _Container>& __lhs, const __btas_normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() < __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator>(const __btas_normal_iterator<_IteratorL, _Container>& __lhs, const __btas_normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() > __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator>(const __btas_normal_iterator<_Iterator, _Container>& __lhs, const __btas_normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() > __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator<=(const __btas_normal_iterator<_IteratorL, _Container>& __lhs, const __btas_normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() <= __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator<=(const __btas_normal_iterator<_Iterator, _Container>& __lhs, const __btas_normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() <= __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline bool
operator>=(const __btas_normal_iterator<_IteratorL, _Container>& __lhs, const __btas_normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() >= __rhs.base(); }

template<typename _Iterator, typename _Container>
inline bool
operator>=(const __btas_normal_iterator<_Iterator, _Container>& __lhs, const __btas_normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() >= __rhs.base(); }

template<typename _IteratorL, typename _IteratorR, typename _Container>
inline typename __btas_normal_iterator<_IteratorL, _Container>::difference_type
operator-(const __btas_normal_iterator<_IteratorL, _Container>& __lhs, const __btas_normal_iterator<_IteratorR, _Container>& __rhs)
{ return __lhs.base() - __rhs.base(); }

template<typename _Iterator, typename _Container>
inline typename __btas_normal_iterator<_Iterator, _Container>::difference_type
operator-(const __btas_normal_iterator<_Iterator, _Container>& __lhs, const __btas_normal_iterator<_Iterator, _Container>& __rhs)
{ return __lhs.base() - __rhs.base(); }

template<typename _Iterator, typename _Container>
inline __btas_normal_iterator<_Iterator, _Container>
operator+(typename __btas_normal_iterator<_Iterator, _Container>::difference_type __n, const __btas_normal_iterator<_Iterator, _Container>& __i)
{ return __btas_normal_iterator<_Iterator, _Container>(__i.base() + __n); }

}; // namespace btas

#endif // __BTAS_ITERATOR_H
