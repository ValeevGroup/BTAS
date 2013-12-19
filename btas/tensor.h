#ifndef __BTAS_TENSOR_H
#define __BTAS_TENSOR_H 1

#include <cassert>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

#include <btas/types.h>
#include <btas/defaults.h>
#include <btas/tensor_traits.h>
#include <btas/array_adaptor.h>

#include <boost/serialization/serialization.hpp>

namespace btas {

  /** BTAS implementation of "dense" tensor class that models \ref labelTWGTensor "TWG.BoxTensor" concept
      @tparam _T element type, Tensor contains values of this type
      @tparam _Range Range type, models \ref labelTWGRange "TWG.Range" concept
      @tparam _Storage Storage type, models \ref labelTWGStorage "TWG.Storage" concept
  */
  template<typename _T,
           class _Range = btas::DEFAULT::range,
           class _Storage = btas::DEFAULT::storage<_T>
           >
  class Tensor {

    public:

      /// value type
      typedef _T value_type;

      /// type of underlying data storage
      typedef _Storage storage_type;

      /// size type
      typedef typename storage_type::size_type size_type;

      // TODO Do we need refinement of Storage iterators???

      /// iterator
      typedef typename storage_type::iterator iterator;

      /// constant iterator
      typedef typename storage_type::const_iterator const_iterator;

      /// type of Range
      typedef _Range range_type;

      /// type of index
      typedef typename _Range::index_type index_type;

      /// TiledArray-specific additions, will go away soon
      typedef Tensor eval_type;

    public:

      /// default constructor
      Tensor () { }

      /// destructor
      ~Tensor () { }

      /// constructor with index extent
      template<typename... _args>
      explicit
      Tensor (const size_type& first, const _args&... rest) :
      range_(range_type(first, rest...))
      {
        array_adaptor<storage_type>::resize(data_, range_.area());
      }

      /// construct from \c range, allocate data, but not initialized
      explicit
      Tensor (const range_type& range) :
      range_(range)
      {
        array_adaptor<storage_type>::resize(data_, range_.area());
      }

      /// construct from \c range object, set all elements to \c v
      explicit
      Tensor (const range_type& range,
              value_type v) :
              range_(range)
      {
        array_adaptor<storage_type>::resize(data_, range_.area());
        std::fill(begin(), end(), v);
      }

      /// copy constructor
      template<class _Tensor, class = typename std::enable_if<is_tensor<_Tensor>::value>::type>
      explicit
      Tensor (const _Tensor& x)
      : range_ (x.range()),
      // TODO this can be optimized to bitewise copy if x::value_type and my value_type are equal, and storage is linear
        data_(x.begin(), x.end())
      {
      }

      /// copy constructor
      explicit
      Tensor (const Tensor& x)
      : range_ (x.range()), data_(x.data_)
      {
      }

      /// copy assignment operator
      // TODO I only know how to do this if _Tensor's range_type is same as mine
      template<class _Tensor, class = typename std::enable_if<is_tensor<_Tensor>::value>::type>
      Tensor&
      operator= (const _Tensor& x)
      {
          range_ = x.range();
          array_adaptor<storage_type>::resize(data_, range_.area());
          std::copy(x.begin(), x.end(), data_.begin());
          return *this;
      }

      /// copy assignment
      Tensor&
      operator= (const Tensor& x)
      {
        range_ = x.range_;
        data_ = x.data_;
        return *this;
      }

      /// move constructor
      explicit
      Tensor (Tensor&& x)
      {
        swap(range_, x.range_);
        swap(data_, x.data_);
      }

      /// move assignment operator
      Tensor&
      operator= (Tensor&& x)
      {
        std::swap(range_, x.range_);
        std::swap(data_, x.data_);
        return *this;
      }

      /// number of indices (tensor rank)
      size_type
      rank () const
      {
        return range_.rank();
      }

      /// \return number of elements
      size_type
      size () const
      {
        return range_.area();
      }

      /// \return range object
      const range_type&
      range() const
      {
        return range_;
      }

      /// test whether storage is empty
      bool
      empty() const
      {
        return range_.area() == 0;
      }

      /// \return const iterator begin
      const_iterator
      begin() const
      {
        return data_.begin();
      }

      /// \return const iterator end
      const_iterator
      end() const
      {
        return data_.end();
      }

      /// \return const iterator begin, even if this is not itself const
      const_iterator
      cbegin() const
      {
        return data_.begin();
      }

      /// \return const iterator end, even if this is not itself const
      const_iterator
      cend() const
      {
        return data_.end();
      }

      /// \return iterator begin
      iterator
      begin()
      {
        return data_.begin();
      }

      /// \return iterator end
      iterator
      end()
      {
        return data_.end();
      }

      /// \return element without range check
      template<typename... _args>
      const value_type&
      operator() (const size_type& first, const _args&... rest) const
      {
        const size_t n = sizeof...(_args) + 1;
        const size_type indexv[n] = {first, static_cast<size_type>(rest)...};
        const index_type index(indexv, indexv+n);
        return data_[ range_.ordinal(index) ];
      }

      /// \return element without range check (rank() == general)
      const value_type&
      operator() (const index_type& index) const
      {
        return data_[range_.ordinal(index)];
      }

      /// access element without range check
      template<typename... _args>
      value_type&
      operator() (const size_type& first, const _args&... rest)
      {
        const size_t n = sizeof...(_args) + 1;
        const size_type indexv[n] = {first, static_cast<size_type>(rest)...};
        const index_type index(indexv, indexv+n);
        return data_[ range_.ordinal(index) ];
      }

      /// access element without range check (rank() == general)
      value_type&
      operator() (const index_type& index)
      {
        return data_[range_.ordinal(index)];
      }
   
      /// \return element without range check
      template<typename... _args>
      const value_type&
      at (const size_type& first, const _args&... rest) const
      {
        const size_t n = sizeof...(_args) + 1;
        const size_type indexv[n] = {first, static_cast<size_type>(rest)...};
        const index_type index(indexv, indexv+n);
        assert( range_.includes(index) );
        return data_[ range_.ordinal(index) ];
      }

      /// \return element without range check (rank() == general)
      const value_type&
      at (const index_type& index) const
      {
        assert( range_.includes(index) );
        return data_[ range_.ordinal(index) ];
      }

      /// access element without range check
      template<typename... _args>
      value_type&
      at (const size_type& first, const _args&... rest)
      {
        const size_t n = sizeof...(_args) + 1;
        const size_type indexv[n] = {first, static_cast<size_type>(rest)...};
        const index_type index(indexv, indexv+n);
        assert( range_.includes(index) );
        return data_[ range_.ordinal(index) ];
      }

      /// access element without range check (rank() == general)
      value_type&
      at (const index_type& index)
      {
        assert( range_.includes(index) );
        return data_[ range_.ordinal(index) ];
      }
   
      /// resize array with range object
      void
      resize (const range_type& range)
      {
        range_ = range;
        array_adaptor<storage_type>::resize(data_, range_.area());
      }

      /// swap this and x
      void
      swap (Tensor& x)
      {
        std::swap(range_, x.range_);
        std::swap(data_, x.data_);
      }

      /// clear all members
      void
      clear()
      {
        range_ = range_type();
        data_ = storage_type();
      }

      //  ========== Finished Public Interface and Its Reference Implementations ==========

      //
      //  Here come Non-Standard members (to be discussed)
      //

      /// addition assignment
      Tensor&
      operator+= (const Tensor& x)
      {
        assert( std::equal(range_.begin(), range_.end(), x.range_.begin()) );
        std::transform(data_.begin(), data_.end(), x.data_.begin(), data_.begin(), std::plus<value_type>());
        return *this;
      }

      /// addition of tensors
      Tensor
      operator+ (const Tensor& x) const
      {
        Tensor y(*this); y += x;
        return y; /* automatically called move semantics */
      }

      /// subtraction assignment
      Tensor&
      operator-= (const Tensor& x)
      {
        assert(
            std::equal(range_.begin(), range_.end(), x.range_.begin()));
        std::transform(data_.begin(), data_.end(), x.data_.begin(), data_.begin(), std::minus<value_type>());
        return *this;
      }

      /// subtraction of tensors
      Tensor
      operator- (const Tensor& x) const
      {
        Tensor y(*this); y -= x;
        return y; /* automatically called move semantics */
      }

      /// \return bare const pointer to the first element of data_
      /// this enables to call BLAS functions
      const value_type*
      data () const
      {
        return data_.data();
      }

      /// \return bare pointer to the first element of data_
      /// this enables to call BLAS functions
      value_type*
      data()
      {
        return data_.data();
      }

      /// fill all elements by val
      void
      fill (const value_type& val)
      {
        std::fill(data_.begin(), data_.end(), val);
      }

      /// generate all elements by gen()
      template<class Generator>
      void
      generate (Generator gen)
      {
          std::generate(data_.begin(), data_.end(), gen);
      }

    private:

      range_type range_;///< range object
      storage_type data_;///< data

  }; // end of Tensor

  /// maps Tensor -> Range
  template <typename _T, typename _Range, typename _Storage>
  btas::Range
  range (const btas::Tensor<_T, _Range, _Storage>& t) {
    return t.range();
  }

  /// maps Tensor -> Range extent
  template <typename _T, typename _Range, typename _Storage>
  btas::Range::extent_type
  extent (const btas::Tensor<_T, _Range, _Storage>& t) {
    return t.range().extent();
  }

} // namespace btas

namespace boost {
namespace serialization {

  /// boost serialization
  template<class Archive, typename _T, class _Storage, class _Range>
  void serialize(Archive& ar, btas::Tensor<_T, _Range, _Storage>& t,
                 const unsigned int version) {
    ar & t.range() & t.stride() & t.data();
  }

}
}

#endif // __BTAS_TENSOR_H
