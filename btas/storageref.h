/*
 * storageref.h
 *
 *  Created on: Dec 27, 2013
 *      Author: evaleev
 */

#ifndef BTAS_STORAGEREF_H_
#define BTAS_STORAGEREF_H_

#include <btas/storage_traits.h>

namespace btas {

  /// Wraps pointer to \c _Storage in _Storage-like interface
  template <typename _Storage>
  class StorageRef {
    public:
      typedef _Storage storage_type;
      typedef typename storage_type::value_type value_type;
      typedef typename storage_type::size_type size_type;
      typedef typename storage_type::iterator iterator;
      typedef typename storage_type::const_iterator const_iterator;

      StorageRef() : begin_(), end_() { end_ = begin_; }
      ~StorageRef() {}

      StorageRef(storage_type& stor) : begin_(stor.begin()), end_(stor.end()) {}
      template <typename Iter1, typename Iter2>
      StorageRef(Iter1 b, Iter2 e) : begin_(b), end_(e) {}
      StorageRef(const StorageRef& other) : begin_(other.begin_), end_(other.end_) {}
      StorageRef& operator=(const StorageRef& other) {
        begin_ = other.begin_;
        end_ = other.end_;
      }
      StorageRef& operator=(storage_type& stor) {
        begin_ = stor.begin();
        end_ = stor.end();
      }
      // no point in move functionality

      value_type& operator[](size_t i) {
        return *(begin_ + i);
      }
      const value_type& operator[](size_t i) const {
        return *(begin_ + i);
      }

      iterator begin() {
        return begin_;
      }
      iterator end() {
        return end_;
      }

    private:
      iterator begin_; // begin
      iterator end_;   // end
  };

  template <typename _Storage>
  class storage_traits<StorageRef<_Storage> > {
    public:
  };
}


#endif /* BTAS_STORAGEREF_H_ */
