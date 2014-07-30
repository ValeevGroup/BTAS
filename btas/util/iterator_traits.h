#ifndef __BTAS_UTIL_ITERATORTRAITS_H_
#define __BTAS_UTIL_ITERATORTRAITS_H_

#include <boost/iterator/iterator_facade.hpp>

namespace btas {

  /// checks whether \c Iterator is a "const iterator", i.e. whether it dereferences (via operator*()) to a reference to a const type
  /// \warning this will not work if \c Iterator::pointer is a proxy type.
  template <typename Iterator> struct is_const_iterator :
    std::conditional<
    std::is_const<typename std::remove_pointer<typename std::iterator_traits<Iterator>::pointer>::type>::value,
    std::true_type,
    std::false_type>::type {
  };

  /// wraps an Iterator into a "const Iterator"
  template <typename Iterator>
  class iterator_const_wrapper : public boost::iterator_facade<
                                          iterator_const_wrapper<Iterator>,
                                          typename std::add_const<typename std::iterator_traits<Iterator>::value_type>::type,
                                          boost::forward_traversal_tag
                                        > {
    public:
      iterator_const_wrapper(const Iterator& iter) : iter_(iter) {}
      operator Iterator() { return iter_; }

    private:
      friend class boost::iterator_core_access;

      bool equal(const iterator_const_wrapper& other) const
      {
        return this->iter_ == other.iter_;
      }

      void increment()
      { ++iter_; }

      void advance(size_t n)
      { iter_ += n; }

      Iterator iter_;

      auto dereference() const -> decltype(*(this->iter_)) { return *iter_; }

  };


  // This metafunction converts const_iterator (or equivalent) to the corresponding iterator
  template <typename ConstIterator>
  struct remove_const_from_iterator;

  /// specializes remove_const_from_iterator<ConstIterator> for iterators wrapped with btas::iterator_const_wrapper
  template <typename Iterator>
  struct remove_const_from_iterator<btas::iterator_const_wrapper<Iterator>> {
      typedef Iterator type;
  };

  /// specializes remove_const_from_iterator<ConstIterator> for pointers
  template <typename T>
  struct remove_const_from_iterator<const T*> {
      typedef const T* type;
  };

  namespace detail {
    template <typename Container>
    Container deduce_container_type_from_const_iterator (typename Container :: const_iterator);
    template <typename Container>
    Container deduce_container_type_from_iterator (typename Container :: iterator);
  };

  /// specializes remove_const_from_iterator<ConstIterator> for const_iterator over a standard-compliant container
  template <typename ConstIterator>
  struct remove_const_from_iterator
  {
      typedef typename decltype (detail::deduce_container_type_from_const_iterator (ConstIterator())) :: iterator type;
  };

}

#endif /* __BTAS_UTIL_ITERATORTRAITS_H_ */
