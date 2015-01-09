/*
 * tensorview_iterator.h
 *
 *  Created on: Dec 28, 2013
 *      Author: evaleev
 */

#ifndef BTAS_TENSORVIEW_ITERATOR_H_
#define BTAS_TENSORVIEW_ITERATOR_H_

#include <cstddef>

#include <btas/storage_traits.h>

namespace btas {

  /// Iterates over values in a range spanned by \c StorageIterator using ordinal values of indices in \c Range

  template <typename Range, typename StorageIterator>
  class TensorViewIterator : public std::iterator<typename std::conditional<btas::is_const_iterator<StorageIterator>::value,
                                                                            std::forward_iterator_tag,
                                                                            std::output_iterator_tag>::type,
                                                  typename std::iterator_traits<StorageIterator>::value_type,
                                                  typename std::iterator_traits<StorageIterator>::difference_type,
                                                  typename std::iterator_traits<StorageIterator>::pointer,
                                                  typename std::iterator_traits<StorageIterator>::reference>
  {
      struct Enabler {};

    public:
      typedef StorageIterator storage_iterator;
      typedef StorageIterator nonconst_storage_iterator;
      typedef std::iterator<typename std::conditional<btas::is_const_iterator<StorageIterator>::value,
                                                      std::forward_iterator_tag,
                                                      std::output_iterator_tag>::type,
                            typename std::iterator_traits<StorageIterator>::value_type,
                            typename std::iterator_traits<StorageIterator>::difference_type,
                            typename std::iterator_traits<StorageIterator>::pointer,
                            typename std::iterator_traits<StorageIterator>::reference> base_type;
      using typename base_type::value_type;
      using typename base_type::pointer;
      using typename base_type::reference;
      using typename base_type::difference_type;
      using typename base_type::iterator_category;

    private:
      typedef typename Range::ordinal_subiterator subiterator;
      typedef typename Range::ordinal_iterator iterator;
      typedef typename iterator::value_type ordinal_type;
      typedef typename Range::index_type index_type;

    public:
      /// Default constructor
      TensorViewIterator() {}
      /// Destructor
      ~TensorViewIterator() {}

      TensorViewIterator(const typename Range::iterator& index_iter,
                         const StorageIterator& storage_begin) :
        iter_(subiterator(std::make_pair(*index_iter,index_iter.range()->ordinal(*index_iter)),index_iter.range())),
        storage_begin_(storage_begin) {}

//      template <typename I = StorageIterator>
//      TensorViewIterator(const typename Range::iterator& index_iter,
//                         const nonconst_storage_iterator& storage_begin,
//                         typename std::enable_if<std::is_const<typename std::iterator_traits<I>::value_type>::value>::type* = 0) :
//        iter_(subiterator(std::make_pair(*index_iter,index_iter.range()->ordinal(*index_iter)),index_iter.range())),
//        // standard const_cast cannot "map" const into nontrivial structures, have to reinterpret here
//        storage_begin_(reinterpret_cast<storage_iterator&>(storage_begin)) {}

      TensorViewIterator(const typename Range::iterator& index_iter,
                         const ordinal_type& ord,
                         const StorageIterator& storage_begin) :
        iter_(subiterator(std::make_pair(*index_iter,ord),index_iter.range())),
        storage_begin_(storage_begin) {}


      TensorViewIterator(const iterator& iter,
                         const StorageIterator& storage_begin) :
        iter_(iter), storage_begin_(storage_begin) {}

      TensorViewIterator(iterator&& iter,
                         const StorageIterator& storage_begin) :
        iter_(iter), storage_begin_(storage_begin) {}

      TensorViewIterator& operator++() {
        ++iter_;
        return *this;
      }

      const reference operator*() const {
        return *(storage_begin_ + *iter_);
      }

      //template <class = typename std::enable_if<not std::is_const<storage_type>::value,Enabler>::type>
      template <typename I = StorageIterator>
      typename std::enable_if<not std::is_const<typename std::iterator_traits<I>::value_type>::value,reference>::type
      operator*() {
        return *(storage_begin_ + *iter_);
      }

      const index_type& index() const {
        return first(*iter_.base());
      }

      template <typename R, typename I>
      friend bool operator==(const TensorViewIterator<R,I>&, const TensorViewIterator<R,I>&);

    private:
      iterator iter_;
      storage_iterator storage_begin_;
  };

  template <typename Range, typename StorageIterator>
  inline bool operator==(const TensorViewIterator<Range,StorageIterator>& i1,
                         const TensorViewIterator<Range,StorageIterator>& i2) {
    return i1.iter_ == i2.iter_;
  }

  template <typename Range, typename StorageIterator>
  inline bool operator!=(const TensorViewIterator<Range,StorageIterator>& i1,
                         const TensorViewIterator<Range,StorageIterator>& i2) {
    return not (i1 == i2);
  }

}


#endif /* BTAS_TENSORVIEW_ITERATOR_H_ */
