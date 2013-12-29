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
  /// If \c _Storage is const, will provide read-access only
  template <typename _Storage>
  class StorageRef {
      struct Enabler {};
    public:
      typedef _Storage storage_type;
      typedef typename std::remove_const<storage_type>::type nonconst_storage_type;
      typedef typename storage_type::value_type value_type;
      typedef typename storage_type::size_type size_type;
      typedef typename std::conditional<std::is_const<storage_type>::value,typename storage_type::const_iterator,typename storage_type::iterator>::type iterator;
      typedef typename storage_type::const_iterator const_iterator;

      StorageRef() : begin_(), end_() { end_ = begin_; }
      ~StorageRef() {}

      template<class = typename std::enable_if<not std::is_const<storage_type>::value,Enabler>::type>
      StorageRef(nonconst_storage_type& stor) : begin_(stor.begin()), end_(stor.end()) {}

      template<class = typename std::enable_if<std::is_const<storage_type>::value,Enabler>::type>
      StorageRef(const nonconst_storage_type& stor) : begin_(stor.cbegin()), end_(stor.cend()) {}

      template <typename Iter1, typename Iter2>
      StorageRef(Iter1 b, Iter2 e) : begin_(b), end_(e) {}
      StorageRef(const StorageRef& other) : begin_(other.begin_), end_(other.end_) {}
      StorageRef& operator=(const StorageRef& other) {
        begin_ = other.begin_;
        end_ = other.end_;
        return *this;
      }

      template<class = typename std::enable_if<not std::is_const<storage_type>::value,Enabler>::type>
      StorageRef& operator=(nonconst_storage_type& stor) {
        begin_ = stor.begin();
        end_ = stor.end();
        return *this;
      }

      template<class = typename std::enable_if<std::is_const<storage_type>::value,Enabler>::type>
      StorageRef& operator=(const nonconst_storage_type& stor) {
        begin_ = stor.cbegin();
        end_ = stor.cend();
        return *this;
      }

      // no point in move functionality

      template<class = typename std::enable_if<not std::is_const<storage_type>::value,Enabler>::type>
      value_type& operator[](size_t i) {
        return *(begin_ + i);
      }
      const value_type& operator[](size_t i) const {
        return *(begin_ + i);
      }

      template<class = typename std::enable_if<not std::is_const<storage_type>::value,Enabler>::type>
      iterator begin() {
        return begin_;
      }
      const_iterator begin() const {
        return begin_;
      }

      template<class = typename std::enable_if<not std::is_const<storage_type>::value,Enabler>::type>
      iterator end() {
        return end_;
      }
      const_iterator end() const {
        return end_;
      }

    private:
      iterator begin_; // begin
      iterator end_;   // end
  }; // StorageRef

}


#endif /* BTAS_STORAGEREF_H_ */
