/*
 * storage_traits.h
 *
 *  Created on: Dec 27, 2013
 *      Author: evaleev
 */

#ifndef BTAS_STORAGE_TRAITS_H_
#define BTAS_STORAGE_TRAITS_H_

#include <valarray>

#include <btas/index_traits.h>

namespace btas {

  /// describes storage traits; user must provide explicit specialization that defined the following types
  /// \code
  /// template <typename _Storage>
  /// struct storage_traits {
  ///   typedef ... /* e.g., typename _Storage::value_type */ value_type;
  ///   typedef ... /* e.g., typename _Storage::pointer */ pointer;
  ///   typedef ... /* e.g., typename _Storage::const_pointer */ const_pointer;
  ///   typedef ... /* e.g., typename _Storage::reference */ reference;
  ///   typedef ... /* e.g., typename _Storage::const_reference */ const_reference;
  ///   typedef ... /* e.g., typename _Storage::size_type */ size_type;
  ///   typedef ... /* e.g., typename _Storage::difference_type */ difference_type;
  ///   typedef ... /* e.g., typename _Storage::iterator */ iterator;
  ///   typedef ... /* e.g., typename _Storage::const_iterator */ const_iterator;
  ///
  ///   template <typename U> rebind_t = ... ; // evaluates to _Storage counterpart storing objects of type U
  ///                                          // e.g. if _Storage is std::vector<T,A> this should be std::vector<U,std::allocator_traits<A>::rebind_alloc<U>>
  /// };
  /// \endcode
  template <typename _Storage>
  struct storage_traits;

  template <typename _T>
  struct storage_traits<_T*> {
      typedef typename std::remove_const<_T>::type value_type;
      typedef _T* pointer;
      typedef typename std::add_const<pointer>::type const_pointer;
      typedef          value_type& reference;
      typedef    const value_type& const_reference;
      typedef size_t size_type;
      typedef ptrdiff_t difference_type;

      typedef pointer iterator;
      typedef const_pointer const_iterator;

      template <typename U> using rebind_t = U*;
  };

  template <typename _T>
  struct storage_traits<_T* const> {
      typedef typename std::remove_const<_T>::type value_type;
      typedef _T* pointer;
      typedef typename std::add_const<pointer>::type const_pointer;
      typedef          value_type& reference;
      typedef    const value_type& const_reference;
      typedef size_t size_type;
      typedef ptrdiff_t difference_type;

      typedef pointer iterator;
      typedef const_pointer const_iterator;

      template <typename U> using rebind_t = U* const;
  };

  template <typename _T>
  struct storage_traits<std::valarray<_T>> {
      typedef _T value_type;
      typedef _T* pointer;
      typedef typename std::add_const<pointer>::type const_pointer;
      typedef _T& reference;
      typedef const _T& const_reference;
      typedef size_t size_type;
      typedef ptrdiff_t difference_type;

      typedef _T* iterator;
      typedef typename std::add_const<_T*>::type const_iterator;

      template <typename U> using rebind_t = std::valarray<U>;
  };

  template <typename _Container>
  struct storage_traits_base_container {
      using value_type = typename _Container::value_type;
      using pointer = typename _Container::pointer;
      using const_pointer = typename _Container::const_pointer;
      using iterator = typename _Container::iterator;
      using const_iterator = typename _Container::const_iterator;
      using size_type = typename _Container::size_type;
      using difference_type = typename _Container::difference_type;
  };

  template <typename _T, std::size_t _N>
  struct storage_traits<std::array<_T, _N>> : public storage_traits_base_container<std::array<_T, _N>> {
      template <typename U> using rebind_t = std::array<U, _N>;
  };

  template <typename _T, typename _Allocator>
  struct storage_traits<std::vector<_T, _Allocator>> : public storage_traits_base_container<std::vector<_T, _Allocator>> {
      template <typename U> using rebind_t = std::vector<U, typename std::allocator_traits<_Allocator>::template rebind_alloc<U>>;
  };

  template <typename _T, typename _Allocator>
  struct storage_traits<varray<_T, _Allocator>> : public storage_traits_base_container<varray<_T, _Allocator>> {
      template <typename U> using rebind_t = varray<U, typename std::allocator_traits<_Allocator>::template rebind_alloc<U>>;
  };

  // specialize to const container; N.B. T* const is not consistent of container<T> const since the latter passes constness onto values
  template <typename _Storage>
  struct storage_traits<_Storage const> {
      using value_type = typename storage_traits<_Storage>::value_type;
      using pointer = typename storage_traits<_Storage>::const_pointer;
      using const_pointer = typename storage_traits<_Storage>::const_pointer;
      using reference = typename storage_traits<_Storage>::const_reference;
      using const_reference = typename storage_traits<_Storage>::const_reference;
      using iterator = typename storage_traits<_Storage>::const_iterator;
      using const_iterator = typename storage_traits<_Storage>::const_iterator;
      using size_type = typename storage_traits<_Storage>::size_type;
      using difference_type = typename storage_traits<_Storage>::difference_type;

      template <typename U> using rebind_t = std::add_const_t<typename storage_traits<_Storage>::template rebind_t<U>>;
  };

  /// test if _Storage conforms to the TWG.Storage concept
  /// in addition to Storage, check extent() member and extent_type
  template<class _Storage>
  class is_storage {
  public:
     static constexpr const bool
     value = has_begin<_Storage>::value &
             has_end<_Storage>::value;
  };


}  // namespace btas


#endif /* BTAS_STORAGE_TRAITS_H_ */
