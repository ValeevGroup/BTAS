/*
 * range.h
 *
 *  Created on: Nov 26, 2013
 *      Author: evaleev
 */

#ifndef BTAS_RANGE_H_
#define BTAS_RANGE_H_

#include <algorithm>
#include <vector>
#include <functional>
#include <numeric>
#include <initializer_list>

#include <btas/varray/varray.h>
#include <btas/range_iterator.h>
#include <btas/array_adaptor.h>
#include <btas/types.h>
#include <btas/type_traits.h>
#include <btas/index_traits.h>
#include <btas/range_traits.h>

/** @addtogroup BTAS_Range

    \section sec_BTAS_Range Range class
    Range implements the Range TWG concept. It supports dense and strided ranges, with fixed (compile-time) and variable (run-time)
    ranks.

    \subsection sec_BTAS_Range_Synopsis Synopsis
    The following will be valid with the reference implementation of Range. This does not belong to the concept specification,
    and not all of these operations will model the concept, but it is useful for discussion; will eventually be moved elsewhere.
    @code
    // Constructors
    Range1 r0;         // empty = {}
    Range1 r1(5);      // [0,5) = {0, 1, 2, 3, 4}
    Range1 r2(2,4);    // [2,4) = {2, 3}
    Range1 r3(1,7,2);  // [1,7) with stride 2 = {1, 3, 5}
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

    template <typename Index = long>
    class Range1d {
      public:
        typedef Index index_type;
        typedef index_type value_type;
        typedef const value_type const_reference_type;

        typedef RangeIterator<index_type, Range1d> const_iterator; ///< Index iterator
        friend class RangeIterator<index_type, Range1d>;

        Range1d(size_t extent = 0ul) :
          lobound_(0), upbound_(extent), stride_(1) {}

        /// [begin, end)
        Range1d(index_type begin, index_type end, index_type stride = 1) :
        lobound_(begin), upbound_(end), stride_(stride) {}

        /// to construct from an initializer list give it as {}, {extent}, {begin,end}, or {begin,end,stride}
        template <typename T> Range1d(std::initializer_list<T> x) : lobound_(0), upbound_(0), stride_(1) {
          assert(x.size() <= 3 //, "Range1d initializer-list constructor requires at most 3 parameters"
                 );
          if (x.size() == 1)
            upbound_ = *x.begin();
          else if (x.size() >= 2) {
            lobound_ = *x.begin();
            upbound_ = *(x.begin()+1);
            if (x.size() == 3)
              stride_ = *(x.begin()+2);
          }
        }

        Range1d(const Range1d& other) :
          lobound_(other.lobound_), upbound_(other.upbound_), stride_(other.stride_)
        { }

        Range1d& operator=(const Range1d& other) {
          lobound_ = other.lobound_;
          upbound_ = other.upbound_;
          stride_ = other.stride_;
          return *this;
        }

        Range1d& operator=(Range1d&& other) {
          lobound_ = other.lobound_;
          upbound_ = other.upbound_;
          stride_ = other.stride_;
          return *this;
        }

        /// to construct from an initializer list give it as {}, {extent}, {begin,end}, or {begin,end,stride}
        template <typename T>
        Range1d& operator=(std::initializer_list<T> x) {
          assert(x.size() <= 3 //, "Range1d initializer-list constructor requires at most 3 parameters"
                 );
          if (x.size() == 0) {
            lobound_ = upbound_ = 0;
            stride_ = 1;
          }
          if (x.size() == 1) {
            lobound_ = 0;
            upbound_ = *x.begin();
            stride_ = 1;
          }
          else if (x.size() >= 2) {
            lobound_ = *x.begin();
            upbound_ = *(x.begin()+1);
            if (x.size() == 3)
              stride_ = *(x.begin()+2);
            else
              stride_ = 1;
          }
          return *this;
        }

        /// \return The rank (number of dimensions) of this range
        /// \throw nothing
        size_t rank() const {
          return 1ul;
        }

        const_reference_type lobound() const { return lobound_; }
        index_type front() const { return lobound_; }
        const_reference_type upbound() const { return upbound_; }
        index_type back() const { return upbound_ - 1; }

        const_reference_type stride() const { return stride_; }

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


      private:
        index_type lobound_;
        index_type upbound_;
        index_type stride_;

        /// Increment the coordinate index \c i in this range

        /// \param[in,out] i The coordinate index to be incremented
        void increment(index_type& i) const {
          i += stride_;
          if(i < upbound_)
            return;
          // if ended up outside the range, set to end
          i = upbound_;
        }


    }; // Range1d
    using Range1 = Range1d<>;

    /// Range1d output operator

    /// \param os The output stream that will be used to print \c r
    /// \param r The range to be printed
    /// \return A reference to the output stream
    template <typename _Index>
    inline std::ostream& operator<<(std::ostream& os, const Range1d<_Index>& r) {
      os << "[" << r.lobound() << "," << r.upbound();
      if (r.stride() != 1ul)
        os << "," << r.stride();
      os << ")";
      return os;
    }


    /// RangeNd data of an N-dimensional tensor
    /// Index rank is a runtime parameter
    template <CBLAS_ORDER _Order = CblasRowMajor,
              typename _Index = btas::varray<long>,
              class = typename std::enable_if<
                is_index<_Index>::value
              >::type>
    class RangeNd {
    public:
      typedef RangeNd Range_; ///< This object type
      typedef _Index index_type; ///< index type
      typedef std::size_t size_type; ///< Size type
      typedef index_type value_type; ///< Range can be viewed as a Container of value_type
      typedef index_type& reference_type;
      typedef const value_type& const_reference_type;
      typedef RangeIterator<index_type, Range_> const_iterator; ///< Index iterator
      typedef const_iterator iterator; ///< interator = const_iterator
      typedef typename std::make_unsigned<index_type>::type extent_type;    ///< Range extent type
      typedef std::size_t ordinal_type; ///< Ordinal type

      friend class RangeIterator<index_type, Range_>;

    private:
      struct Enabler {};

      template <typename Index1, typename Index2>
      void init(const Index1& lobound, const Index2& upbound) {
        using btas::rank;
        auto n = rank(lobound);
        if (n == 0) return;

        typedef typename common_signed_type<typename Index1::value_type, typename Index2::value_type>::type ctype;

        std::size_t volume = 1ul;
        lobound_ = array_adaptor<index_type>::construct(n);
        upbound_ = array_adaptor<index_type>::construct(n);
        weight_ = array_adaptor<extent_type>::construct(n);

        // Compute range data
        if (_Order == CblasRowMajor) {
          for(int i = n - 1; i >= 0; --i) {
            auto li = *(lobound.begin() + i);
            auto ui = *(upbound.begin() + i);
            assert(static_cast<ctype>(li) <= static_cast<ctype>(ui));
            lobound_[i] = li;
            upbound_[i] = ui;
            weight_[i] = volume;
            volume *= (upbound_[i] - lobound_[i]);
          }
        }
        else {
          for(auto i = 0; i != n; ++i) {
            auto li = *(lobound.begin() + i);
            auto ui = *(upbound.begin() + i);
            assert(static_cast<ctype>(li) <= static_cast<ctype>(ui));
            lobound_[i] = li;
            upbound_[i] = ui;
            weight_[i] = volume;
            volume *= (upbound_[i] - lobound_[i]);
          }
        }
      }

      void init() {
        auto n = rank();
        if (n == 0) return;

        std::size_t volume = 1ul;
        weight_ = array_adaptor<extent_type>::construct(n);

        // Compute range data
        if (_Order == CblasRowMajor) {
          for(int i = n - 1; i >= 0; --i) {
            auto li = *(lobound_.begin() + i);
            auto ui = *(upbound_.begin() + i);
            assert(li <= ui);
            weight_[i] = volume;
            volume *= (ui - li);
          }
        }
        else {
          for(auto i = 0; i != n; ++i) {
            auto li = *(lobound_.begin() + i);
            auto ui = *(upbound_.begin() + i);
            assert(li <= ui);
            weight_[i] = volume;
            volume *= (ui - li);
          }
        }
      }

      template <typename Index1, typename Index2, typename Extent>
      void init(const Index1& lobound, const Index2& upbound, const Extent& weight) {
        using btas::rank;
        auto n = rank(lobound);
        if (n == 0) return;

        typedef typename common_signed_type<typename Index1::value_type, typename Index2::value_type>::type ctype;

        for(auto i = 0; i != n; ++i) {
          auto li = *(lobound.begin() + i);
          auto ui = *(upbound.begin() + i);
          assert(static_cast<ctype>(li) <= static_cast<ctype>(ui));
        }

        lobound_ = array_adaptor<index_type>::construct(n);
        std::copy(lobound.begin(), lobound.end(), lobound_.begin());
        upbound_ = array_adaptor<index_type>::construct(n);
        std::copy(upbound.begin(), upbound.end(), upbound_.begin());
        weight_ = array_adaptor<extent_type>::construct(n);
        std::copy(weight.begin(), weight.end(), weight_.begin());

      }

      template <typename Extent>
      void init(const Extent& extent) {
        using btas::rank;
        auto n = rank(extent);
        if (n == 0) return;

        // now I know the rank
        lobound_ = array_adaptor<index_type>::construct(n, 0);

        std::size_t volume = 1ul;
        upbound_ = array_adaptor<index_type>::construct(n);
        weight_ = array_adaptor<extent_type>::construct(n);

        // Compute range data
        if (_Order == CblasRowMajor) {
          for(int i = n - 1; i >= 0; --i) {
            upbound_[i] = *(extent.begin() + i);
            weight_[i] = volume;
            volume *= upbound_[i];
          }
        }
        else {
          for(auto i = 0; i != n; ++i) {
            upbound_[i] = *(extent.begin() + i);
            weight_[i] = volume;
            volume *= upbound_[i];
          }
        }

      }

    public:

      /// Default constructor

      /// Construct a range with size and dimensions equal to zero.
      RangeNd() :
        lobound_(), upbound_(), weight_()
      { }

      /// Constructor defined by the upper and lower bounds

      /// \tparam Index1 An array type convertible to \c index_type
      /// \tparam Index2 An array type convertible to \c index_type
      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      template <typename Index1, typename Index2>
      RangeNd(const Index1& lobound, const Index2& upbound,
              typename std::enable_if<btas::is_index<Index1>::value && btas::is_index<Index2>::value, Enabler>::type = Enabler()) :
        lobound_(), upbound_(), weight_()
      {
        using btas::rank;
        auto n = rank(lobound);
        assert(n == rank(upbound));
        init(lobound, upbound);
      }

      /// "Move" constructor defined by the upper and lower bounds

      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      RangeNd(index_type&& lobound, index_type&& upbound) :
        lobound_(lobound), upbound_(upbound), weight_()
      {
        using btas::rank;
        auto n = rank(lobound);
        assert(n == rank(upbound));
        init();
      }

      /// Constructor defined by the upper and lower bounds, and the axes weights

      /// \tparam Index1 An array type convertible to \c index_type
      /// \tparam Index2 An array type convertible to \c index_type
      /// \tparam Extent An array type convertible to \c extent_type
      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      /// \param weight The axes weights of the N-dimensional range
      template <typename Index1, typename Index2, typename Extent>
      RangeNd(const Index1& lobound, const Index2& upbound, const Extent& weight,
              typename std::enable_if<btas::is_index<Index1>::value &&
                                      btas::is_index<Index2>::value &&
                                      btas::is_index<Extent>::value, Enabler>::type = Enabler()) :
        lobound_(), upbound_(), weight_()
      {
        using btas::rank;
        auto n = rank(lobound);
        assert(n == rank(upbound));
        assert(n == rank(weight));
        init(lobound, upbound, weight);
      }

      /// "Move" constructor defined by the upper and lower bounds, and the axes weights

      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      /// \param weight The axes weights of the N-dimensional range
      RangeNd(index_type&& lobound, index_type&& upbound, extent_type&& weight) :
        lobound_(lobound), upbound_(upbound), weight_(weight)
      {
        using btas::rank;
        auto n = rank(lobound);
        assert(n == rank(upbound));
        assert(n == rank(weight));
        for(auto i = 0; i != n; ++i) {
          auto li = *(lobound.begin() + i);
          auto ui = *(upbound.begin() + i);
          assert(li <= ui);
        }
      }

      /// Range constructor from extent

      /// \tparam Extent An array type convertible to \c extent_type
      /// \param extent An array with the extent of each dimension
      template <typename Extent,
                class = typename std::enable_if<btas::is_index<Extent>::value>::type>
      RangeNd(const Extent& extent) :
        lobound_(), upbound_(), weight_()
      {
        init(extent);
      }

      /// Range constructor from a pack of extents for each dimension

      /// \tparam _extent0 An integer
      /// \tparam _extents A pack of integers
      /// \param extent0 The extent of first dimension (0)
      /// \param sizes A pack of sizes for dimensions 1+
      template<typename _extent0, typename... _extents, class = typename std::enable_if<std::is_integral<_extent0>::value>::type>
      explicit RangeNd(const _extent0& extent0, const _extents&... extents) :
      lobound_(), upbound_(), weight_()
      {
        typedef typename std::common_type<_extent0, typename extent_type::value_type>::type common_type;
        // make initializer_list
        auto range_extent = {static_cast<common_type>(extent0), static_cast<common_type>(extents)...};
        init(range_extent);
      }

      /// to construct from an initializer list give it as {extent0, extent1, ... extentN}
      template <typename T>
      RangeNd(std::initializer_list<T> extents) :
      lobound_(), upbound_(), weight_()
      {
        init(extents);
      }

      /// to construct from an initializer list give it as {extent0, extent1, ... extentN}
      template <typename T1, typename T2>
      RangeNd(std::initializer_list<T1> lobound, std::initializer_list<T2> upbound) :
      lobound_(), upbound_(), weight_()
      {
        assert(lobound.size() == upbound.size());
        init(lobound, upbound);
      }

      /// Copy Constructor

      /// \param other The range to be copied
      RangeNd(const Range_& other) :
        lobound_(other.lobound_), upbound_(other.upbound_), weight_(other.weight_)
      {
      }

      /// copy constructor from another instantiation of Range
      template <CBLAS_ORDER Order,
                typename Index>
      RangeNd (const RangeNd<Order,Index>& x)
      {
          init(x.lobound(), x.upbound());
      }

      /// Move Constructor

      /// \param other The range to be moved
      RangeNd(Range_&& other) :
        lobound_(other.lobound_), upbound_(other.upbound_), weight_(other.weight_)
      {
      }


      /// Destructor
      ~RangeNd() { }

      /// Copy assignment operator

      /// \param other The range to be copied
      /// \return A reference to this object
      /// \throw std::bad_alloc When memory allocation fails.
      Range_& operator=(const Range_& other) {
        lobound_ = other.lobound_;
        upbound_ = other.upbound_;
        weight_ = other.weight_;

        return *this;
      }

      /// Access a particular dimension of Range

      Range1d<typename index_type::value_type> dim(size_t d) const {
        return Range1d<typename index_type::value_type>(*(lobound_.begin()+d), *(upbound_.begin()+d));
      }

      /// Resize range to a new upper and lower bound

      /// This can be used to avoid memory allocation
      /// \tparam Index An array type
      /// \param lobound The lower bounds of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      template <typename Index>
      typename std::enable_if<btas::is_index<Index>::value, Range_&>::type
      resize(const Index& lobound, const Index& upbound) {
        using btas::rank;
        auto n = rank(lobound);
        assert(n == rank(upbound));
        init(lobound, upbound);
        return *this;
      }

      /// This can be used to avoid memory allocation
      /// \tparam Index An array type
      /// \param lobound The lower bounds of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      template <typename Extent>
      typename std::enable_if<btas::is_index<Extent>::value, Range_&>::type
      resize(const Extent& extent) {
        init(extent);
        return *this;
      }

      /// Rank accessor

      /// \return The rank (number of dimensions) of this range
      /// \throw nothing
      size_t rank() const {
        using btas::rank;
        return rank(lobound_);
      }

      /// Range lobound coordinate accessor

      /// \return A \c size_array that contains the lower bound of this range
      /// \throw nothing
      const_reference_type lobound() const { return lobound_; }

      /// Range lobound coordinate accessor

      /// \return A \c size_array that contains the first index in this range
      /// \throw nothing
      index_type front() const { return lobound_; }

      /// Range upbound coordinate accessor

      /// \return A \c size_array that contains the upper bound of this range
      /// \throw nothing
      const_reference_type upbound() const {
        return upbound_;
      }

      /// Range size accessor

      /// \return A \c extent_type that contains the extent of each dimension
      /// \throw nothing
      extent_type extent() const {
        extent_type ex = array_adaptor<extent_type>::construct(rank());
        for(auto i=0; i<rank(); ++i)
          ex[i] = upbound_[i] - lobound_[i];
        return ex;
      }

      /// Range weight accessor

      /// \return A \c size_array that contains the strides of each dimension
      /// \throw nothing
      const extent_type& weight() const { return weight_; }

      /// Range volume accessor

      /// \return The total number of elements in the range.
      /// \throw nothing
      size_type area() const {
        if (rank()) {
          const extent_type ex = extent();
          return std::accumulate(ex.begin(), ex.end(), 1ul, std::multiplies<size_type>());
        }
        else
          return 0;
      }

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

      /// calculate the ordinal index of \c i

      /// Convert an index to its ordinal.
      /// \tparam Index A coordinate index type (array type)
      /// \param index The index to be converted to an ordinal index
      /// \return The ordinal index of \c index
      /// \throw When \c index is not included in this range.
      template <typename Index>
      typename std::enable_if<btas::is_index<Index>::value, size_type>::type
      ordinal(const Index& index) const {
        using btas::rank;
        assert(rank(index) == this->rank());
        assert(this->includes(index));
        size_type o = 0;
        const auto end = this->rank();
        for(auto i = 0ul; i < end; ++i)
          o += (index[i] - lobound_[i]) * weight_[i];

        return o;
      }

      /// Check the coordinate to make sure it is within the range.

      /// \tparam Index The coordinate index array type
      /// \param index The coordinate index to check for inclusion in the range
      /// \return \c true when \c i \c >= \c lobound and \c i \c < \c f, otherwise
      /// \c false
      /// \throw TildedArray::Exception When the dimension of this range is not
      /// equal to the size of the index.
      template <typename Index>
      typename std::enable_if<btas::is_index<Index>::value, bool>::type
      includes(const Index& index) const {
        using btas::rank;
        assert(rank(index) == this->rank());
        const auto end = this->rank();
        for(auto i = 0ul; i < end; ++i)
          if((index[i] < lobound_[i]) || (index[i] >= upbound_[i]))
            return false;

        return true;
      }


      void swap(Range_& other) {
        std::swap(lobound_, other.lobound_);
        std::swap(upbound_, other.upbound_);
        std::swap(weight_, other.weight_);
      }

    private:

      /// Increment the coordinate index \c i in this range

      /// \param[in,out] i The coordinate index to be incremented
      void increment(index_type& i) const {

        if (_Order == CblasRowMajor) {
          for(int d = int(rank()) - 1; d >= 0; --d) {
            // increment coordinate
            ++i[d];

            // break if done
            if(i[d] < upbound_[d])
              return;

            // Reset current index to lobound value.
            i[d] = lobound_[d];
          }
        }
        else { // col-major
          for(auto d = 0ul; d != rank(); ++d) {
            // increment coordinate
            ++i[d];

            // break if done
            if(i[d] < upbound_[d])
              return;

            // Reset current index to lobound value.
            i[d] = lobound_[d];
          }
        }

        // if the current location is outside the range, make it equal to range end iterator
        std::copy(upbound_.begin(), upbound_.end(), i.begin());
      }

#if 0
      /// Advance the coordinate index \c i by \c n in this range

      /// \param[in,out] i The coordinate index to be advanced
      /// \param n The distance to advance \c i
      void advance(index& i, std::ptrdiff_t n) const {
        const size_type o = ord(i) + n;
        i = idx(o);
      }

      /// Compute the distance between the coordinate indices \c first and \c last

      /// \param first The lobounding position in the range
      /// \param last The ending position in the range
      /// \return The difference between first and last, in terms of range positions
      std::ptrdiff_t distance_to(const index& first, const index& last) const {
        TA_ASSERT(includes(first));
        TA_ASSERT(includes(last));
        return ord(last) - ord(first);
      }
#endif

      index_type lobound_; ///< range origin
      index_type upbound_; ///< range extent

      // optimization details
      extent_type weight_; ///< Dimension weights (strides)
    }; // class RangeNd

    using Range = RangeNd<>;

    /// Exchange the values of the give two ranges.
    template <CBLAS_ORDER _Order,
              typename _Index>
    inline void swap(RangeNd<_Order,_Index>& r0, RangeNd<_Order,_Index>& r1) { // no throw
      r0.swap(r1);
    }


    /// Range equality comparison

    /// \param r1 The first range to be compared
    /// \param r2 The second range to be compared
    /// \return \c true when \c r1 represents the same range as \c r2, otherwise
    /// \c false.
    template <CBLAS_ORDER _Order,
              typename _Index>
    inline bool operator ==(const RangeNd<_Order,_Index>& r1, const RangeNd<_Order,_Index>& r2) {
      return ((r1.lobound() == r2.lobound()) && (r1.extent() == r2.extent()));
    }

    /// Range inequality comparison

    /// \param r1 The first range to be compared
    /// \param r2 The second range to be compared
    /// \return \c true when \c r1 does not represent the same range as \c r2,
    /// otherwise \c false.
    template <CBLAS_ORDER _Order,
              typename _Index>
    inline bool operator !=(const RangeNd<_Order,_Index>& r1, const RangeNd<_Order,_Index>& r2) {
      return ! operator ==(r1, r2);
    }

    /// Transposes a Range

    /// permutes the axes using permutation \c p = {p[0], p[1], ... }; for example, if \c lobound() initially returned
    /// {lb[0], lb[1], ... }, after this call \c lobound() will return {lb[p[0]], lb[p[1]], ...} .
    /// \param perm an array specifying permutation of the axes
    template <CBLAS_ORDER _Order,
              typename _Index,
              typename AxisPermutation,
              class = typename std::enable_if<btas::is_index<AxisPermutation>::value>::type>
    RangeNd<_Order, _Index> transpose(const RangeNd<_Order, _Index>& r,
                                      const AxisPermutation& perm)
    {
      const auto rank = r.rank();
      auto lb = r.lobound();
      auto ub = r.upbound();
      auto wt = r.weight();

      typedef typename RangeNd<_Order, _Index>::index_type index_type;
      typedef typename RangeNd<_Order, _Index>::extent_type extent_type;
      index_type lobound, upbound;
      extent_type weight;
      lobound = array_adaptor<index_type>::construct(rank);
      upbound = array_adaptor<index_type>::construct(rank);
      weight = array_adaptor<extent_type>::construct(rank);

      std::for_each(perm.begin(), perm.end(), [&](const typename AxisPermutation::value_type& i){
        const auto pi = *(perm.begin() + i);
        *(lobound.begin()+i) = *(lb.begin() + pi);
        *(upbound.begin()+i) = *(ub.begin() + pi);
        *(weight.begin()+i) = *(wt.begin() + pi);
      });

      return RangeNd<_Order, _Index>(std::move(lobound), std::move(upbound), std::move(weight));
    }


    /// Range output operator

    /// \param os The output stream that will be used to print \c r
    /// \param r The range to be printed
    /// \return A reference to the output stream
    template <CBLAS_ORDER _Order,
              typename _Index>
    inline std::ostream& operator<<(std::ostream& os, const RangeNd<_Order,_Index>& r) {
      os << "[";
      array_adaptor<_Index>::print(r.lobound(), os);
      os << ",";
      array_adaptor<_Index>::print(r.upbound(), os);
      os << ")_" << (_Order == CblasRowMajor ? "R" : "C");
      os << ":" << r.weight();
      return os;
    }

    namespace boost {
    namespace serialization {

      /// boost serialization
      template<class Archive, CBLAS_ORDER _Order,
               typename _Index>
      void serialize(Archive& ar, btas::RangeNd<_Order, _Index>& t,
                     const unsigned int version) {
        ar & t.lobound() & t.upbound() & t.weight();
      }

    }
    }

    template <CBLAS_ORDER _Order,
              typename _Index>
    class boxrange_iteration_order< btas::RangeNd<_Order,_Index> > {
      public:
        enum {row_major = boxrange_iteration_order<void>::row_major,
              other = boxrange_iteration_order<void>::other,
              column_major = boxrange_iteration_order<void>::column_major};

        static constexpr int value = (_Order == CblasRowMajor) ? row_major : column_major;
    };
}

#endif /* BTAS_RANGE_H_ */
