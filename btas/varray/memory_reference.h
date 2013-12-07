#ifndef __BTAS_MEMORY_REFERENCE_H
#define __BTAS_MEMORY_REFERENCE_H 1

#include <iterator>
#include <type_traits>

#include <btas/varray/btas_iterator.h>

namespace btas {

/// container skeleton
template<typename _T>
struct memory_reference {

   typedef _T value_type;

   typedef value_type& reference;

   typedef const value_type& const_reference;

   typedef value_type* pointer;

   typedef const value_type* const_pointer;

   typedef __btas_normal_iterator<pointer, memory_reference> iterator;

   typedef __btas_normal_iterator<const_pointer, memory_reference> const_iterator;

   typedef std::reverse_iterator<iterator> reverse_iterator;

   typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

   typedef typename std::iterator_traits<iterator>::difference_type difference_type;

   typedef size_t size_type;

   //
   // member variables
   //

   pointer _M_start; ///< pointer to the first

   pointer _M_finish; ///< pointer to the end

   //
   // constructors
   //

   memory_reference ()
   : _M_start (nullptr), _M_finish (nullptr)
   { }

   memory_reference (pointer first, pointer last)
   : _M_start (first), _M_finish (last)
   { }

   memory_reference (const memory_reference& x)
   : _M_start (x._M_start), _M_finish (x._M_finish)
   { }

   template<class _Container>
   memory_reference (_Container& x)
   : _M_start (nullptr), _M_finish (nullptr)
   {
      static_assert(std::is_same<typename _Container::value_type, value_type>::value, "Error: mismatched value_type's");
      if (!x.empty()) {
         _M_start (x.data());
         _M_finish = _M_start + x.size();
      }
   }
   
   memory_reference& operator= (const memory_reference& x)
   {
      _M_start = x._M_start;
      _M_finish = x._M_finish;
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

   void swap (memory_reference& x)
   {
      std::swap (_M_start, x._M_start);
      std::swap (_M_finish, x._M_finish);
   }

};

}; // namespace btas

#endif // __BTAS_MEMORY_REFERENCE_H
