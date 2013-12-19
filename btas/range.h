/*
 * range.h
 *
 *  Created on: Nov 26, 2013
 *      Author: evaleev
 */

#ifndef RANGE_H_
#define RANGE_H_

#include <algorithm>
#include <vector>
#include <functional>

#include <btas/varray/varray.h>
#include <btas/range_iterator.h>
#include <btas/array_adaptor.h>
#include <btas/types.h>
#include <btas/index_traits.h>

//#include <TiledArray/range.h>

/** @addtogroup BTAS_Range

    \section sec_BTAS_Range Range class
    Range implements the Range TWG concept. It supports dense and strided ranges, with fixed (compile-time) and variable (run-time)
    ranks.

    \subsection sec_BTAS_Range_Synopsis Synopsis
    The following will be valid with the reference implementation of Range. This does not belong to the concept specification,
    and not all of these operations will model the concept, but it is useful for discussion; will eventually be moved elsewhere.
    @code
    // Constructors
    Range<1> r0;         // empty = {}
    Range<1> r1(5);      // [0,5) = {0, 1, 2, 3, 4}
    Range<1> r2(2,4);    // [2,4) = {2, 3}
    Range<1> r3(1,7,2);  // [1,7) with stride 2 = {1, 3, 5}
    assert(r3.rank() == 1);
    Range x(r2,r3);   // r1 x r2 = { {2,1}, {2,3}, {2,5}, {4,1}, {4,3}, {4,5} }
    assert(x.rank() == 2);

    // Operations
    std::cout << x.area() << std::endl;  // will print "6"

    // Iteration
    for(auto& v: r3) {
      std::cout << v << " "; // will print "1 3 5 "
    }
    @endcode
*/

namespace btas {
  //using TiledArray::Range;

    namespace detail {

      template <typename Index, typename WeightArray, typename StartArray>
      inline std::size_t calc_ordinal(const Index& index, const WeightArray& weight, const StartArray& lobound) {
        // Check that the dimensions of the arrays are equal.
        const std::size_t n = rank(index);
        assert(rank(weight) == n);
        assert(rank(lobound) == n);

        // Compute ordinal
        std::size_t o = 0ul;
        for(std::size_t i = 0ul; i < n; ++i)
          o += (index[i] - lobound[i]) * weight[i];

        return o;
      }

    }  // namespace detail

    class Range1 {
      public:
        Range1(std::size_t extent = 0ul) :
          begin_(0ul), end_(extent), stride_(1ul) {}

        Range1(std::size_t begin, std::size_t end, std::size_t stride_ = 1ul) :
        begin_(begin), end_(end), stride_(1ul) {}

        Range1(const Range1& other) :
          begin_(other.begin_), end_(other.end_), stride_(other.stride_)
        { }

        Range1& operator=(const Range1& other) {
          begin_ = other.begin_;
          end_ = other.end_;
          stride_ = other.stride_;
          return *this;
        }

      private:
        std::size_t begin_;
        std::size_t end_;
        std::size_t stride_;
    };

    /// Range data of an N-dimensional tensor
    /// Index rank is a runtime parameter
    template <CBLAS_ORDER _Order = CblasRowMajor,
              typename _Index = btas::varray<std::size_t> >
    class Range {
    public:
      typedef Range Range_; ///< This object type
      typedef std::size_t size_type; ///< Size type
      typedef _Index index_type; ///< Coordinate index type
      typedef index_type extent_type;    ///< Range extent type
      typedef std::size_t ordinal_type; ///< Ordinal type
      typedef RangeIterator<index_type, Range_> const_iterator; ///< Index iterator
      friend class RangeIterator<index_type, Range_>;

    private:
      struct Enabler {};

      template <typename Index>
      void init(const Index& lobound, const Index& upbound) {
        using btas::rank;
        auto n = rank(lobound);
        if (n == 0) return;

        std::size_t volume = 1ul;
        lobound_ = array_adaptor<index_type>::construct(n);
        extent_ = array_adaptor<extent_type>::construct(n);
        weight_ = array_adaptor<extent_type>::construct(n);

        // Compute range data
        if (_Order == CblasRowMajor) {
          for(int i = n - 1; i >= 0; --i) {
            assert(lobound[i] <= upbound[i]);
            lobound_[i] = lobound[i];
            extent_[i] = upbound[i] - lobound[i];
            weight_[i] = volume;
            volume *= extent_[i];
          }
        }
        else {
          for(auto i = 0; i != n; ++i) {
            assert(lobound[i] <= upbound[i]);
            lobound_[i] = lobound[i];
            extent_[i] = upbound[i] - lobound[i];
            weight_[i] = volume;
            volume *= extent_[i];
          }
        }
      }

      template <typename Index>
      void init(const Index& extent) {
        using btas::rank;
        auto n = rank(extent);
        if (n == 0) return;

        std::size_t volume = 1ul;
        lobound_ = array_adaptor<index_type>::construct(n);
        extent_ = array_adaptor<extent_type>::construct(n);
        weight_ = array_adaptor<extent_type>::construct(n);

        // Compute range data
        if (_Order == CblasRowMajor) {
          for(int i = n - 1; i >= 0; --i) {
            assert(extent[i] > 0);
            lobound_[i] = 0ul;
            extent_[i] = extent[i];
            weight_[i] = volume;
            volume *= extent_[i];
          }
        }
        else {
          for(auto i = 0; i != n; ++i) {
            assert(extent[i] > 0);
            lobound_[i] = 0ul;
            extent_[i] = extent[i];
            weight_[i] = volume;
            volume *= extent_[i];
          }
        }

      }

    public:

      /// Default constructor

      /// Construct a range with size and dimensions equal to zero.
      Range() :
        lobound_(), extent_(), weight_()
      { }

      /// Constructor defined by an upper and lower bound

      /// \tparam Index An array type
      /// \param lobound The lower bounds of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      /// \throw TiledArray::Exception When the size of \c lobound is not equal to
      /// that of \c upbound.
      /// \throw TiledArray::Exception When lobound[i] >= upbound[i]
      /// \throw std::bad_alloc When memory allocation fails.
      template <typename Index>
      Range(const Index& lobound, const Index& upbound,
            typename std::enable_if<btas::is_index<Index>::value, Enabler>::type = Enabler()) :
        lobound_(), extent_(), weight_()
      {
        using btas::rank;
        auto n = rank(lobound);
        assert(n == rank(upbound));
        init(lobound, upbound);
      }

      /// Range constructor from size array

      /// \tparam SizeArray An array type
      /// \param extent An array with the extent of each dimension
      /// \throw std::bad_alloc When memory allocation fails.
      template <typename SizeArray>
      Range(const SizeArray& extent) :
        lobound_(), extent_(), weight_()
      {
        using btas::rank;
        extent_ = array_adaptor<index_type>::construct(rank(extent));
        std::copy(extent.begin(), extent.end(), extent_.begin());
        init(extent);
      }

      /// Range constructor from a pack of sizes for each dimension

      /// \tparam _size0 A
      /// \tparam _sizes A pack of unsigned integers
      /// \param sizes The size of dimensions 0
      /// \param sizes A pack of sizes for dimensions 1+
      /// \throw std::bad_alloc When memory allocation fails.
      template<typename... _sizes>
      explicit Range(const size_type& size0, const _sizes&... sizes) :
      lobound_(), extent_(), weight_()
      {
        const size_type n = sizeof...(_sizes) + 1;
        size_type range_extent[n] = {size0, static_cast<size_type>(sizes)...};
        init(range_extent);
      }

      /// Copy Constructor

      /// \param other The range to be copied
      /// \throw std::bad_alloc When memory allocation fails.
      Range(const Range_& other) :
        lobound_(other.lobound_), extent_(other.extent_), weight_(other.weight_)
      {
      }

      /// Destructor
      ~Range() { }

      /// Copy assignment operator

      /// \param other The range to be copied
      /// \return A reference to this object
      /// \throw std::bad_alloc When memory allocation fails.
      Range_& operator=(const Range_& other) {
        lobound_ = other.lobound_;
        extent_ = other.extent_;
        weight_ = other.weight_;

        return *this;
      }

      /// Dimension accessor

      /// \return The rank (number of dimensions) of this range
      /// \throw nothing
      size_t rank() const {
        using btas::rank;
        return rank(lobound_);
      }

      /// Range lobound coordinate accessor

      /// \return A \c size_array that contains the lower bound of this range
      /// \throw nothing
      const index_type& lobound() const { return lobound_; }

      /// Range upbound coordinate accessor

      /// \return A \c size_array that contains the upper bound of this range
      /// \throw nothing
      index_type upbound() const {
        index_type up = array_adaptor<index_type>::construct(rank());
        for(auto i=0; i<rank(); ++i)
          up[i] = lobound_[i] + extent_[i];
        return up;
      }

      /// Range size accessor

      /// \return A \c extent_type that contains the extent of each dimension
      /// \throw nothing
      extent_type extent() const { return extent_; }

      /// Range weight accessor

      /// \return A \c size_array that contains the strides of each dimension
      /// \throw nothing
      const extent_type& weight() const { return weight_; }

      /// Range volume accessor

      /// \return The total number of elements in the range.
      /// \throw nothing
      size_type area() const {
        if (rank())
          return _Order == CblasRowMajor ?
              weight_[0] * extent_[0] :
              weight_[rank()-1] * extent_[rank()-1];
        else
          return 0;
      }

#if 0
      /// Index iterator factory

      /// The iterator dereferences to an index. The order of iteration matches
      /// the data layout of a dense tensor.
      /// \return An iterator that holds the lobound element index of a tensor
      /// \throw nothing
      const_iterator begin() const { return const_iterator(lobound_, this); }

      /// Index iterator factory

      /// The iterator dereferences to an index. The order of iteration matches
      /// the data layout of a dense tensor.
      /// \return An iterator that holds the upbound element index of a tensor
      /// \throw nothing
      const_iterator end() const { return const_iterator(upbound_, this); }

      /// Check the coordinate to make sure it is within the range.

      /// \tparam Index The coordinate index array type
      /// \param index The coordinate index to check for inclusion in the range
      /// \return \c true when \c i \c >= \c lobound and \c i \c < \c f, otherwise
      /// \c false
      /// \throw TildedArray::Exception When the dimension of this range is not
      /// equal to the size of the index.
      template <typename Index>
      typename madness::disable_if<std::is_integral<Index>, bool>::type
      includes(const Index& index) const {
        TA_ASSERT(detail::size(index) == dim());
        const unsigned int end = dim();
        for(unsigned int i = 0ul; i < end; ++i)
          if((index[i] < lobound_[i]) || (index[i] >= upbound_[i]))
            return false;

        return true;
      }

      /// Check the ordinal index to make sure it is within the range.

      /// \param i The ordinal index to check for inclusion in the range
      /// \return \c true when \c i \c >= \c 0 and \c i \c < \c volume
      /// \throw nothing
      template <typename Ordinal>
      typename madness::enable_if<std::is_integral<Ordinal>, bool>::type
      includes(Ordinal i) const {
        return include_ordinal_(i);
      }

      /// Resize range to a new upper and lower bound

      /// \tparam Index An array type
      /// \param lobound The lower bounds of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      /// \throw TiledArray::Exception When the size of \c lobound is not equal to
      /// that of \c upbound.
      /// \throw TiledArray::Exception When lobound[i] >= upbound[i]
      /// \throw std::bad_alloc When memory allocation fails.
      template <typename Index>
      Range_& resize(const Index& lobound, const Index& upbound) {
        const size_type n = detail::size(lobound);
        TA_ASSERT(n == detail::size(upbound));

        // Reallocate memory for range arrays
        realloc_arrays(n);
        if(n > 0ul)
          compute_range_data(n, lobound, upbound);
        else
          volume_ = 0ul;

        return *this;
      }

      /// calculate the ordinal index of \c i

      /// This function is just a pass-through so the user can call \c ord() on
      /// a template parameter that can be a coordinate index or an integral.
      /// \param index Ordinal index
      /// \return \c index (unchanged)
      /// \throw When \c index is not included in this range
      size_type ord(const size_type index) const {
        TA_ASSERT(includes(index));
        return index;
      }

      /// calculate the ordinal index of \c i

      /// Convert a coordinate index to an ordinal index.
      /// \tparam Index A coordinate index type (array type)
      /// \param index The index to be converted to an ordinal index
      /// \return The ordinal index of \c index
      /// \throw When \c index is not included in this range.
      template <typename Index>
      typename madness::disable_if<std::is_integral<Index>, size_type>::type
      ord(const Index& index) const {
        TA_ASSERT(detail::size(index) == dim());
        TA_ASSERT(includes(index));
        size_type o = 0;
        const unsigned int end = dim();
        for(unsigned int i = 0ul; i < end; ++i)
          o += (index[i] - lobound_[i]) * weight_[i];

        return o;
      }

      /// alias to ord<Index>(), to conform with the TWG spec \sa ord()
      template <typename Index>
      typename madness::disable_if<std::is_integral<Index>, size_type>::type
      ordinal(const Index& index) const {
        return ord<Index>(index);
      }

      /// calculate the coordinate index of the ordinal index, \c index.

      /// Convert an ordinal index to a coordinate index.
      /// \param index Ordinal index
      /// \return The index of the ordinal index
      /// \throw TiledArray::Exception When \c index is not included in this range
      /// \throw std::bad_alloc When memory allocation fails
      index idx(size_type index) const {
        // Check that o is contained by range.
        TA_ASSERT(includes(index));

        // Construct result coordinate index object and allocate its memory.
        Range_::index result;
        result.reserve(dim());

        // Compute the coordinate index of o in range.
        for(std::size_t i = 0ul; i < dim(); ++i) {
          const size_type s = index / weight_[i]; // Compute the size of result[i]
          index %= weight_[i];
          result.push_back(s + lobound_[i]);
        }

        return result;
      }

      /// calculate the index of \c i

      /// This function is just a pass-through so the user can call \c idx() on
      /// a template parameter that can be an index or a size_type.
      /// \param i The index
      /// \return \c i (unchanged)
      template <typename Index>
      typename madness::disable_if<std::is_integral<Index>, const index&>::type
      idx(const Index& i) const {
        TA_ASSERT(includes(i));
        return i;
      }

      template <typename Archive>
      typename madness::enable_if<madness::archive::is_input_archive<Archive> >::type
      serialize(const Archive& ar) {
        // Get number of dimensions
        size_type n = 0ul;
        ar & n;

        // Get range data
        realloc_arrays(n);
        ar & madness::archive::wrap(lobound_.data(), n * 4ul) & volume_;
      }

      template <typename Archive>
      typename madness::enable_if<madness::archive::is_output_archive<Archive> >::type
      serialize(const Archive& ar) const {
        const size_type n = dim();
        ar & n & madness::archive::wrap(lobound_.data(), n * 4ul) & volume_;
      }

#endif

      void swap(Range_& other) {
        std::swap(lobound_, other.lobound_);
        std::swap(extent_, other.extent_);
        std::swap(weight_, other.weight_);
      }

    private:

#if 0
      /// Check that a signed integral value is include in this range

      /// \tparam Index A signed integral type
      /// \param i The ordinal index to check
      /// \return \c true when <tt>i >= 0</tt> and <tt>i < volume_</tt>, otherwise
      /// \c false.
      template <typename Index>
      typename madness::enable_if<std::is_signed<Index>, bool>::type
      include_ordinal_(Index i) const { return (i >= Index(0)) && (i < Index(volume_)); }

      /// Check that an unsigned integral value is include in this range

      /// \tparam Index An unsigned integral type
      /// \param i The ordinal index to check
      /// \return \c true when  <tt>i < volume_</tt>, otherwise \c false.
      template <typename Index>
      typename madness::disable_if<std::is_signed<Index>, bool>::type
      include_ordinal_(Index i) const { return i < volume_; }

      /// Increment the coordinate index \c i in this range

      /// \param[in,out] i The coordinate index to be incremented
      /// \throw TiledArray::Exception When the dimension of i is not equal to
      /// that of this range
      /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
      void increment(index& i) const {
        TA_ASSERT(includes(i));
        for(int d = int(dim()) - 1; d >= 0; --d) {
          // increment coordinate
          ++i[d];

          // break if done
          if(i[d] < upbound_[d])
            return;

          // Reset current index to lobound value.
          i[d] = lobound_[d];
        }

        // if the current location was set to lobound then it was at the end and
        // needs to be reset to equal upbound.
        std::copy(upbound_.begin(), upbound_.end(), i.begin());
      }

      /// Advance the coordinate index \c i by \c n in this range

      /// \param[in,out] i The coordinate index to be advanced
      /// \param n The distance to advance \c i
      /// \throw TiledArray::Exception When the dimension of i is not equal to
      /// that of this range
      /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
      void advance(index& i, std::ptrdiff_t n) const {
        TA_ASSERT(includes(i));
        const size_type o = ord(i) + n;
        TA_ASSERT(includes(o));
        i = idx(o);
      }

      /// Compute the distance between the coordinate indices \c first and \c last

      /// \param first The lobounding position in the range
      /// \param last The ending position in the range
      /// \return The difference between first and last, in terms of range positions
      /// \throw TiledArray::Exception When the dimension of \c first or \c last
      /// is not equal to that of this range
      /// \throw TiledArray::Exception When \c first or \c last is outside this range
      std::ptrdiff_t distance_to(const index& first, const index& last) const {
        TA_ASSERT(includes(first));
        TA_ASSERT(includes(last));
        return ord(last) - ord(first);
      }
#endif

      index_type lobound_; ///< range origin
      extent_type extent_; ///< range extent

      // optimization details
      extent_type weight_; ///< Dimension weights (strides)
    }; // class Range

    /// Exchange the values of the give two ranges.
    template <CBLAS_ORDER _Order,
              typename _Index>
    inline void swap(Range<_Order,_Index>& r0, Range<_Order,_Index>& r1) { // no throw
      r0.swap(r1);
    }


    /// Range equality comparison

    /// \param r1 The first range to be compared
    /// \param r2 The second range to be compared
    /// \return \c true when \c r1 represents the same range as \c r2, otherwise
    /// \c false.
    template <CBLAS_ORDER _Order,
              typename _Index>
    inline bool operator ==(const Range<_Order,_Index>& r1, const Range<_Order,_Index>& r2) {
      return ((r1.lobound() == r2.lobound()) && (r1.extent() == r2.extent()));
    }

    /// Range inequality comparison

    /// \param r1 The first range to be compared
    /// \param r2 The second range to be compared
    /// \return \c true when \c r1 does not represent the same range as \c r2,
    /// otherwise \c false.
    template <CBLAS_ORDER _Order,
              typename _Index>
    inline bool operator !=(const Range<_Order,_Index>& r1, const Range<_Order,_Index>& r2) {
      return ! operator ==(r1, r2);
    }

    /// Range output operator

    /// \param os The output stream that will be used to print \c r
    /// \param r The range to be printed
    /// \return A reference to the output stream
    template <CBLAS_ORDER _Order,
              typename _Index>
    inline std::ostream& operator<<(std::ostream& os, const Range<_Order,_Index>& r) {
      os << "[ ";
      array_adaptor<_Index>::print(r.lobound(), os);
      os << ", ";
      array_adaptor<_Index>::print(r.upbound(), os);
      os << " )_" << (_Order == CblasRowMajor ? "R" : "C");
      return os;
    }

}

#endif /* RANGE_H_ */
