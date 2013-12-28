/*
 * tensorview_iterator.h
 *
 *  Created on: Dec 28, 2013
 *      Author: evaleev
 */

#ifndef BTAS_TENSORVIEW_ITERATOR_H_
#define BTAS_TENSORVIEW_ITERATOR_H_

#include <cstddef>

#include <btas/storageref.h>

namespace btas {

  /// Iterates over elements of \c Storage using ordinal values of indices in \c Range

  template <typename Range, typename Storage>
  class TensorViewIterator {
    public:
      typedef StorageRef<Storage> storageref_type;
      typedef typename storageref_type::value_type value_type; ///< Iterator value type
      typedef value_type& reference; ///< Iterator reference type
      typedef const reference const_reference; ///< Iterator reference type
      typedef value_type* pointer; ///< Iterator pointer type
      typedef std::output_iterator_tag iterator_category; /// Iterator category tag
      typedef std::ptrdiff_t difference_type; ///< Iterator difference type

      typedef typename Range::const_iterator index_iterator;
      typedef typename index_iterator::value_type index_type;

      /// Default constructor
      TensorViewIterator() {}
      /// Destructor
      ~TensorViewIterator() {}

      TensorViewIterator(const index_iterator& iter,
                         Storage& storage) :
        iter_(iter), storageref_(storage) {}

      TensorViewIterator(index_iterator&& iter,
                         Storage& storage) :
        iter_(iter), storageref_(storage) {}

      TensorViewIterator& operator++() {
        ++iter_;
        return *this;
      }

      const_reference operator*() const {
        return *(storageref_.begin() + iter_.range()->ordinal(*iter_));
      }

      reference operator*() {
        return *(storageref_.begin() + iter_.range()->ordinal(*iter_));
      }

      const index_type& index() const {
        return *iter_;
      }

      template <typename R, typename S>
      friend bool operator==(const TensorViewIterator<R,S>&, const TensorViewIterator<R,S>&);

    private:
      index_iterator iter_;
      storageref_type storageref_;
  };

  template <typename Range, typename Storage>
  inline bool operator==(const TensorViewIterator<Range,Storage>& i1,
                         const TensorViewIterator<Range,Storage>& i2) {
    return i1.iter_ == i2.iter_;
  }

  template <typename Range, typename Storage>
  inline bool operator!=(const TensorViewIterator<Range,Storage>& i1,
                         const TensorViewIterator<Range,Storage>& i2) {
    return not (i1 == i2);
  }

}


#endif /* BTAS_TENSORVIEW_ITERATOR_H_ */
