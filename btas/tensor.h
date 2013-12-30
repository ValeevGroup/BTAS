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
           class _Storage = btas::DEFAULT::storage<_T>,
           class = typename std::enable_if<std::is_same<_T, typename _Storage::value_type>::value>::type
          >
  class Tensor {

    public:

      /// value type
      typedef _T value_type;

      /// type of underlying data storage
      typedef _Storage storage_type;

      /// size type
      typedef typename storage_type::size_type size_type;

      /// element iterator
      typedef typename storage_type::iterator iterator;

      /// constant element iterator
      typedef typename storage_type::const_iterator const_iterator;

      /// type of Range
      typedef _Range range_type;

      /// type of index
      typedef typename _Range::index_type index_type;

    private:
      struct Enabler {};

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
      range_(range.lobound(), range.upbound())
      {
        array_adaptor<storage_type>::resize(data_, range_.area());
      }

      /// construct from \c range object, set all elements to \c v
      explicit
      Tensor (const range_type& range,
              value_type v) :
              range_(range.lobound(), range.upbound())
      {
        array_adaptor<storage_type>::resize(data_, range_.area());
        std::fill(begin(), end(), v);
      }

      /// construct from \c range and \c storage
      explicit
      Tensor (const range_type& range, const storage_type& storage) :
      range_(range.lobound(), range.upbound()), data_(storage)
      {
      }

      /// move-construct from \c range and \c storage
      explicit
      Tensor (range_type&& range, storage_type&& storage) :
      range_(range.ordinal(*range(begin())) == 0 ? range : range_type(range.lobound(), range.upbound())),
      data_(storage)
      {
      }

      /// copy constructor
      /// It will accept Tensors and TensorViews
      template<class _Tensor, class = typename std::enable_if<is_boxtensor<_Tensor>::value>::type>
      Tensor (const _Tensor& x)
      : range_ (x.range().lobound(), x.range().upbound()),
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
      template<class _Tensor, class = typename std::enable_if<is_boxtensor<_Tensor>::value>::type>
      Tensor&
      operator= (const _Tensor& x)
      {
          range_ = range_type(x.range().lobound(), x.range().upbound());
          array_adaptor<storage_type>::resize(data_, range_.area());
          std::copy(x.begin(), x.end(), data_.begin());
          return *this;
      }

      /// copy assignment operator
      template<class _Tensor, class = typename std::enable_if<is_boxtensor<_Tensor>::value>::type>
      Tensor&
      operator= (_Tensor&& x)
      {
          range_ = range_type(x.range().lobound(), x.range().upbound());
          data_ = x.storage();
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
        std::swap(range_, x.range_);
        std::swap(data_, x.data_);
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

      /// \param d dimension
      /// \return subrange for dimension \d
      const Range1d<typename index_type::value_type>
      range(size_t d) const
      {
        return range_.range(d);
      }

      /// \return range's extent object
      typename range_type::extent_type
      extent() const
      {
        return range_.extent();
      }

      /// \return extent of range along dimension \c d
      typename range_type::extent_type::value_type
      extent(size_t d) const
      {
        return range_.extent(d);
      }

      /// \return storage object
      const storage_type&
      storage() const
      {
        return data_;
      }

      /// \return storage object
      storage_type&
      storage()
      {
        return data_;
      }


      /// test whether Tensor is empty
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
      template<typename index0, typename... _args>
      typename std::enable_if<std::is_integral<index0>::value, const value_type&>::type
      operator() (const index0& first, const _args&... rest) const
      {
        typedef typename common_signed_type<index0, typename index_type::value_type>::type ctype;
        auto indexv = {static_cast<ctype>(first), static_cast<ctype>(rest)...};
        index_type index = array_adaptor<index_type>::construct(indexv.size());
        std::copy(indexv.begin(), indexv.end(), index.begin());
        return data_[ range_.ordinal(index) ];
      }

      /// \return element without range check (rank() == general)
      template <typename Index>
      typename std::enable_if<is_index<Index>::value, const value_type&>::type
      operator() (const Index& index) const
      {
        return data_[range_.ordinal(index)];
      }

      /// access element without range check
      template<typename index0, typename... _args>
      typename std::enable_if<std::is_integral<index0>::value, value_type&>::type
      operator() (const index0& first, const _args&... rest)
      {
        typedef typename common_signed_type<index0, typename index_type::value_type>::type ctype;
        auto indexv = {static_cast<ctype>(first), static_cast<ctype>(rest)...};
        index_type index = array_adaptor<index_type>::construct(indexv.size());
        std::copy(indexv.begin(), indexv.end(), index.begin());
        return data_[ range_.ordinal(index) ];
      }

      /// access element without range check (rank() == general)
      template <typename Index>
      typename std::enable_if<is_index<Index>::value, value_type&>::type
      operator() (const Index& index)
      {
        return data_[range_.ordinal(index)];
      }
   
      /// \return element without range check
      template<typename index0, typename... _args>
      typename std::enable_if<std::is_integral<index0>::value, const value_type&>::type
      at (const index0& first, const _args&... rest) const
      {
        typedef typename common_signed_type<index0, typename index_type::value_type>::type ctype;
        auto indexv = {static_cast<ctype>(first), static_cast<ctype>(rest)...};
        index_type index = array_adaptor<index_type>::construct(indexv.size());
        std::copy(indexv.begin(), indexv.end(), index.begin());
        assert( range_.includes(index) );
        return data_[ range_.ordinal(index) ];
      }

      /// \return element without range check (rank() == general)
      template <typename Index>
      typename std::enable_if<is_index<Index>::value, const value_type&>::type
      at (const Index& index) const
      {
        assert( range_.includes(index) );
        return data_[ range_.ordinal(index) ];
      }

      /// access element without range check
      template<typename index0, typename... _args>
      typename std::enable_if<std::is_integral<index0>::value, value_type&>::type
      at (const index0& first, const _args&... rest)
      {
        typedef typename common_signed_type<index0, typename index_type::value_type>::type ctype;
        auto indexv = {static_cast<ctype>(first), static_cast<ctype>(rest)...};
        index_type index = array_adaptor<index_type>::construct(indexv.size());
        std::copy(indexv.begin(), indexv.end(), index.begin());
        assert( range_.includes(index) );
        return data_[ range_.ordinal(index) ];
      }

      /// access element without range check (rank() == general)
      template <typename Index>
      typename std::enable_if<is_index<Index>::value, value_type&>::type
      at (const Index& index)
      {
        assert( range_.includes(index) );
        return data_[ range_.ordinal(index) ];
      }
   
      /// resize array with range object
      template <typename Range>
      void
      resize (const Range& range, typename std::enable_if<is_boxrange<Range>::value,Enabler>::type = Enabler())
      {
        range_ = range;
        array_adaptor<storage_type>::resize(data_, range_.area());
      }

      /// resize array with extent object
      template <typename Extent>
      void
      resize (const Extent& extent, typename std::enable_if<is_index<Extent>::value,Enabler>::type = Enabler())
      {
        range_ = range_type(extent);
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

  template <typename _T, typename _Range, typename _Storage>
  auto cbegin(const btas::Tensor<_T, _Range, _Storage>& x) -> decltype(x.cbegin()) {
    return x.cbegin();
  }
  template <typename _T, typename _Range, typename _Storage>
  auto cend(const btas::Tensor<_T, _Range, _Storage>& x) -> decltype(x.cbegin()) {
    return x.cend();
  }

  /// maps Tensor -> Range
  template <typename _T, typename _Range, typename _Storage>
  auto
  range (const btas::Tensor<_T, _Range, _Storage>& t) -> decltype(t.range()) {
    return t.range();
  }

  /// maps Tensor -> Range extent
  template <typename _T, typename _Range, typename _Storage>
  auto
  extent (const btas::Tensor<_T, _Range, _Storage>& t) -> decltype(t.range().extent()) {
    return t.range().extent();
  }

  /// Tensor stream output operator

  /// prints Tensor in row-major form. To be implemented elsewhere using slices.
  /// \param os The output stream that will be used to print \c t
  /// \param t The Tensor to be printed
  /// \return A reference to the output stream
  template <typename _T, typename _Range, typename _Storage>
  std::ostream& operator<<(std::ostream& os, const btas::Tensor<_T, _Range, _Storage>& t) {
    os << "Tensor:\n  Range: " << t.range() << std::endl;
    return os;
  }

} // namespace btas

namespace boost {
namespace serialization {

  /// boost serialization
  template<class Archive, typename _T, class _Storage, class _Range>
  void serialize(Archive& ar, btas::Tensor<_T, _Range, _Storage>& t,
                 const unsigned int version) {
    ar & t.range() & t.storage();
  }

}
}

#endif // __BTAS_TENSOR_H
