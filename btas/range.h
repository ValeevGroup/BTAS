/*
 * range.h
 *
 *  Created on: Nov 26, 2013
 *      Author: evaleev
 */

#ifndef BTAS_RANGE_H_
#define BTAS_RANGE_H_

#include <btas/fwd.h>

#include <btas/serialization.h>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <vector>

#ifndef BTAS_HAS_BOOST_ITERATOR
#error \
    "BTAS cannot be used without Boost.Iterator; add Boost dir to the include path"
#else
#include <boost/iterator/transform_iterator.hpp>
#endif

#include <btas/defaults.h>
#include <btas/range_iterator.h>
#include <btas/array_adaptor.h>
#include <btas/types.h>
#include <btas/type_traits.h>
#include <btas/index_traits.h>
#include <btas/range_traits.h>
#include <btas/ordinal.h>
#include <btas/util/functional.h>

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
        typedef const value_type const_reference;

        typedef RangeIterator<index_type, Range1d> const_iterator; ///< Index iterator
        typedef const_iterator iterator; ///< interator = const_iterator
        friend class RangeIterator<index_type, Range1d>;

        Range1d(size_t extent = 0ul) :
          lobound_(0), upbound_(extent), stride_(1) {}

        /// [begin, end)
        Range1d(index_type begin, index_type end, index_type stride = 1) :
        lobound_(begin), upbound_(end), stride_(stride) {
          assert(stride_ != 0);
        }

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
          assert(stride_ != 0);
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
        constexpr size_t rank() const {
          return 1ul;
        }

        const_reference lobound() const { return lobound_; }
        index_type front() const { return lobound_; }
        const_reference upbound() const { return upbound_; }
        index_type back() const { return upbound_ - 1; }

        const_reference stride() const { return stride_; }

        /// Size of Range1d is the number of elements encountered in iteration from begin to end.
        size_t size() const {
          return (upbound_ - lobound_) / stride_;
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

        /// Increment the coordinate index \c i in this range

        /// \param[in,out] i The coordinate index to be incremented
        void increment(index_type& i) const {
          i += stride_;
          if (not_past_end(i))
            return;
          // if ended up outside the range, set to end
          i = upbound_;
        }

      private:
        index_type lobound_;
        index_type upbound_;
        index_type stride_;

        bool not_past_end(const index_type& i) const {
          if (stride_ > 0)
            return i < upbound_;
          else // stride_ < 0
            return i > upbound_;
        }

    }; // Range1d
    using Range1 = Range1d<>;

    /// Merges 2 Range1d objects
    template <typename _Index>
    Range1d<_Index> merge(const Range1d<_Index>& r1,
                          const Range1d<_Index>& r2) {
      assert(r1.stride() == r2.stride());
      assert((r2.lobound() - r1.lobound()) % r1.stride() == 0);
      return Range1d<_Index>{r1.lobound(), r2.upbound(), r1.stride()};
    }

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

    /// Range1d equality operator

    /// comparison of two Range1d objects, with potentially different index types
    /// \tparam _Index1
    /// \tparam _Index2
    /// \param r1 the first, Range1d<_Index1>, object
    /// \param r2 the second, Range1d<_Index1>, object
    /// returns true if \c r1 and \c r2 have identical lobound, upbound, and stride
    template <typename _Index1, typename _Index2>
    bool operator==(const Range1d<_Index1>& r1,
                    const Range1d<_Index2>& r2) {
      return r1.lobound() == r2.lobound() && r1.upbound() == r2.upbound() && r1.stride() == r2.stride();
    }

    /// Range1d inequality operator

    /// comparison of two Range1d objects, with potentially different index types
    /// \tparam _Index1
    /// \tparam _Index2
    /// \param r1 the first, Range1d<_Index1>, object
    /// \param r2 the second, Range1d<_Index1>, object
    /// returns false if \c r1 and \c r2 have identical lobound, upbound, and stride
    template <typename _Index1, typename _Index2>
    bool operator!=(const Range1d<_Index1>& r1,
                    const Range1d<_Index2>& r2) {
      return !operator==(r1,r2);
    }

    /// Range1d congruence test

    /// two Range1d objects are congruent if their sizes are equal
    /// \tparam _Index1
    /// \tparam _Index2
    /// \param r1 the first, Range1d<_Index1>, object
    /// \param r2 the second, Range1d<_Index1>, object
    /// returns true if \c r1 and \c r2 have identical sizes
    template <typename _Index1, typename _Index2>
    bool congruent(const Range1d<_Index1>& r1,
                    const Range1d<_Index2>& r2) {
      return r1.size() == r2.size();
    }

    /// convenient to iterate over dimensions according to \c Order
    template <blas::Layout Order = blas::Layout::RowMajor>
    Range1
    dim_range(size_t ndim) {
      if (Order == blas::Layout::RowMajor)
        return Range1(ndim-1,-1,-1);
      if (Order == blas::Layout::ColMajor)
        return Range1(0,ndim,1);
      assert(false); // unreachable
      return Range1();
    }

    /// BaseRangeNd is a <a href="http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern">CRTP</a>
    /// base for implementations of N-dimensional Ranges.

    /**
     * BaseRangeNd defines a box in the index space, and the iteration order on it.
     * The iteration order depends on the blas::Layout parameter (ordering of dimensions).
     * It implements most of the \ref sec_TWG_Range_Concept_Range_Box "TWG.BoxRange" concept, except it does
     * not define ordinals.
     *
     * \tparam _Derived implementation of Range, to be derived from \c BaseRangeNd as \c public \c BaseRangeNd<Derived>
     *
     */
    template <typename _Derived>
    class BaseRangeNd {
    public:

      const static blas::Layout order = range_traits<_Derived>::order;
      typedef typename range_traits<_Derived>::index_type index_type; ///< index type
      typedef typename std::make_unsigned<index_type>::type extent_type;    ///< Range extent type
      typedef std::size_t size_type; ///< Size type

      typedef typename index_type::value_type index_element_type;
      typedef typename extent_type::value_type extent_element_type;

      typedef index_type value_type; ///< Range can be viewed as a Container of value_type
      typedef index_type& reference;
      typedef const value_type& const_reference;

      // index iterator
      typedef RangeIterator<index_type, _Derived>  iterator;         ///< Index iterator
      typedef iterator const_iterator; ///< Index interator = Index const_iterator

      friend class RangeIterator<index_type, _Derived>;
      friend _Derived;

    private:
      struct Enabler {};

      template <typename Index1, typename Index2>
      void init(const Index1& lobound, const Index2& upbound) {
        using btas::rank;
        auto n = rank(lobound);
        if (n == 0) {
          lobound_ = array_adaptor<index_type>::construct(0);
          upbound_ = array_adaptor<index_type>::construct(0);
          extent_ = array_adaptor<extent_type>::construct(0);
          return;
        }
        validate(lobound, upbound);

        lobound_ = array_adaptor<index_type>::construct(n);
        std::copy(std::begin(lobound), std::end(lobound), std::begin(lobound_));
        upbound_ = array_adaptor<index_type>::construct(n);
        std::copy(std::begin(upbound), std::end(upbound), std::begin(upbound_));
        extent_ = array_adaptor<extent_type>::construct(n);
        std::transform(std::begin(lobound), std::end(lobound), std::begin(upbound), std::begin(extent_),
                       [](index_element_type l, index_element_type u) -> extent_element_type {
          return u - l;
        });
      }

      template <typename Index1, typename Index2>
      void validate(const Index1& lobound, const Index2& upbound) {
#ifndef NDEBUG
        using btas::rank;
        auto n = rank(lobound);
        assert(n == rank(upbound));

        typedef typename common_signed_type<typename Index1::value_type, typename Index2::value_type>::type ctype;
        for(decltype(n) i = 0; i != n; ++i) {
          auto li = *(std::begin(lobound) + i);
          auto ui = *(std::begin(upbound) + i);
          assert(static_cast<ctype>(li) <= static_cast<ctype>(ui));
        }
#endif
      }

    protected:

      /// Default constructor

      /// Construct an uninitialized range
     /// \note this is a rank-0 range if \c rank(index_type) return value is
     ///       non-constexpr
     BaseRangeNd() : lobound_(), upbound_(), extent_() {}

     /// Constructor defined by the upper and lower bounds

     /// \tparam Index1 An array type convertible to \c index_type
     /// \tparam Index2 An array type convertible to \c index_type
     /// \param lobound The lower bound of the N-dimensional range
     /// \param upbound The upper bound of the N-dimensional range
     template <typename Index1, typename Index2>
     BaseRangeNd(const Index1& lobound, const Index2& upbound,
                 typename std::enable_if<btas::is_index<Index1>::value &&
                                             btas::is_index<Index2>::value,
                                         Enabler>::type = Enabler()) {
       validate(lobound, upbound);
       init(lobound, upbound);
      }

      /// "Move" constructor defined by the upper and lower bounds

      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      BaseRangeNd(index_type&& lobound, index_type&& upbound) :
        lobound_(lobound), upbound_(upbound)
      {
        validate(lobound, upbound);
        extent_ = array_adaptor<extent_type>::construct(rank());
        std::transform(
            std::begin(lobound), std::end(lobound), std::begin(upbound),
            std::begin(extent_),
            [](index_element_type l,
               index_element_type u) -> extent_element_type { return u - l; });
      }

      /// Range constructor from a pack of extents for each dimension

      /// \tparam _extent0 An integer
      /// \tparam _extents A pack of integers
      /// \param extent0 The extent of first dimension (0)
      /// \param sizes A pack of sizes for dimensions 1+
      template<typename _extent0, typename... _extents, class = typename std::enable_if<std::is_integral<_extent0>::value>::type>
      explicit BaseRangeNd(const _extent0& extent0, const _extents&... extents)
      {
        typedef typename std::common_type<_extent0, typename extent_type::value_type>::type common_type;
        // make initializer_list
        auto range_extent = {static_cast<common_type>(extent0), static_cast<common_type>(extents)...};
        index_type lb = array_adaptor<index_type>::construct(range_extent.size(), 0);
        init(lb, range_extent);
      }

      /// to construct from an initializer list give it as {extent0, extent1, ... extentN}
      template <typename T>
      BaseRangeNd(std::initializer_list<T> extents)
      {
        index_type lb = array_adaptor<index_type>::construct(extents.size(), 0);
        init(lb, extents);
      }

      /// to construct from an initializer list give it as {extent0, extent1, ... extentN}
      template <typename T1, typename T2>
      BaseRangeNd(std::initializer_list<T1> lobound, std::initializer_list<T2> upbound)
      {
        assert(lobound.size() == upbound.size());
        init(lobound, upbound);
      }

      /// Copy Constructor

      /// \param other The range to be copied
      BaseRangeNd(const BaseRangeNd& other) :
        lobound_(other.lobound_), upbound_(other.upbound_), extent_(other.extent_)
      {
      }

      /// copy constructor from another instantiation of Range
      template <class Derived>
      BaseRangeNd (const BaseRangeNd<Derived>& x)
      {
          init(x.lobound(), x.upbound());
      }

      /// Move Constructor

      /// \param other The range to be moved
      BaseRangeNd(BaseRangeNd&& other) :
        lobound_(std::move(other.lobound_)),
        upbound_(std::move(other.upbound_)),
        extent_(std::move(other.extent_))
      {
      }


      /// Destructor
      ~BaseRangeNd() = default;

      /// Copy assignment operator

      /// \param other The range to be copied
      /// \return A reference to this object
      /// \throw std::bad_alloc When memory allocation fails.
      BaseRangeNd& operator=(const BaseRangeNd& other) {
        lobound_ = other.lobound_;
        upbound_ = other.upbound_;
        extent_ = other.extent_;

        return *this;
      }

      /// Move assignment operator

      /// \param other The range to be moved
      /// \return A reference to this object
      BaseRangeNd& operator=(BaseRangeNd&& other) {
        lobound_ = std::move(other.lobound_);
        upbound_ = std::move(other.upbound_);
        extent_ = std::move(other.extent_);

        return *this;
      }

      /// swaps the contents of \c *this with \c other
      void swap(BaseRangeNd& other) noexcept {
        using std::swap;
        swap(lobound_, other.lobound_);
        swap(upbound_, other.upbound_);
        swap(extent_, other.extent_);
      }

     public:

      /// Access a particular subrange of Range

      /// returns the Range1 corresponding to the dimension \c d
      /// \param d the dimension index
      Range1d<typename index_type::value_type> range(size_t d) const {
        return Range1d<typename index_type::value_type>(*(std::begin(lobound_)+d), *(std::begin(upbound_)+d));
      }

      /// Range lobound coordinate accessor

      /// \return A \c size_array that contains the lower bound of this range
      /// \throw nothing
      const_reference lobound() const { return lobound_; }

      const index_element_type* lobound_data() const {
        return std::data(lobound_);
      }

      /// Range lobound coordinate accessor

      /// \return A \c size_array that contains the first index in this range
      /// \throw nothing
      index_type front() const { return lobound_; }

      /// Range upbound coordinate accessor

      /// \return A \c size_array that contains the upper bound of this range
      /// \throw nothing
      const_reference upbound() const {
        return upbound_;
      }

      const index_element_type* upbound_data() const {
        return std::data(upbound_);
      }

      /// Rank accessor

      /// \return The rank (number of dimensions) of this range
      /// \throw nothing
      //constexpr auto rank() const -> decltype(btas::rank(this->lobound())) {
      constexpr size_t rank() const {
        using btas::rank;
        return rank(lobound_);
      }

      /// Range size accessor

      /// \return A \c extent_type that contains the extent of each dimension
      /// \throw nothing
      const extent_type& extent() const {
        return extent_;
//        extent_type ex = array_adaptor<extent_type>::construct(rank());
//        for(size_t i=0; i<rank(); ++i)
//          ex[i] = upbound_[i] - lobound_[i];
//        return ex;
      }

      const extent_element_type* extent_data() const {
        return std::data(extent_);
      }

      /// \return The extent of the nth dimension
      typename extent_type::value_type
      extent(size_t n) const {
          return extent_[n];
      }

      /// Range volume accessor

      /// \return The total number of elements in the range.
      /// \throw nothing
      size_type area() const {
        if (rank()) {
          return std::accumulate(std::begin(extent_), std::end(extent_), 1ul, std::multiplies<size_type>());
        }
        else
          return 0;
      }

      /// An alias for area()
      size_type volume() const {
        return area();
      }

      /// Index iterator factory

      /// The iterator dereferences to an index. The order of iteration matches
      /// the data layout of a dense tensor.
      /// \return An iterator that holds the lobound element index of a tensor
      /// \throw nothing
      const_iterator begin() const {
        return const_iterator(lobound_, static_cast<const _Derived*>(this)); }

      /// Index iterator factory

      /// The iterator dereferences to an index. The order of iteration matches
      /// the data layout of a dense tensor.
      /// \return An iterator that holds the upbound element index of a tensor
      /// \throw nothing
      const_iterator end() const { return const_iterator(upbound_, static_cast<const _Derived*>(this)); }

      /// Increment index \c i in this range

      /// \param[in,out] i The coordinate index to be incremented
      void increment(index_type& i) const {

        for(auto d: dim_range<order>(rank())) {
          // increment coordinate
          ++i[d];

          // break if done
          if(i[d] < upbound_[d])
            return;

          // Reset current index to lobound value.
          i[d] = lobound_[d];
        }

        // if the current location is outside the range, make it equal to range end iterator
        std::copy(std::begin(upbound_), std::end(upbound_), std::begin(i));
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
        assert(includes(first));
        assert(includes(last));
        return ord(last) - ord(first);
      }
#endif

      /// Check the index to make sure it is within the range.

      /// \tparam Index An array type
      /// \param index The index to check for inclusion in the range
      /// \return \c true when \c i \c >= \c lobound and \c i \c < \c f, otherwise
      /// \c false
      /// equal to the size of the index.
      template <typename Index>
      typename std::enable_if<btas::is_index<Index>::value, bool>::type
      includes(const Index& index, typename std::enable_if<btas::is_index<Index>::value>::type* = 0) const {
        using btas::rank;
        assert(rank(index) == this->rank());
        const auto end = this->rank();
        for(size_t i = 0; i < end; ++i)
          if((index[i] < lobound_[i]) || (index[i] >= upbound_[i]))
            return false;

        return true;
      }

    private:

      /// Validates that the index is in the Range
      /// \tparam Index A coordinate index type (array type)
      /// \param index The index to be converted to an ordinal index
      /// \return The ordinal index of \c index
      /// \throw When \c index is not included in this range.
      template <typename Index>
      typename std::enable_if<btas::is_index<Index>::value, void>::type
      validate_index(const Index& index) const {
        using btas::rank;
        assert(rank(index) == this->rank());
        assert(this->includes(index));
      }

    private:
      index_type lobound_; ///< range lower bound
      index_type upbound_; ///< range upper bound
      extent_type extent_;  ///< range extent

    }; // class BaseRangeNd

    /// RangeNd extends BaseRangeNd to compute ordinals, as specified by \c _Ordinal .
    /// It conforms to the \ref sec_TWG_Range_Concept_Range_Box "TWG.BoxRange" concept.
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal
             >
    class RangeNd : public BaseRangeNd< RangeNd<_Order,_Index, _Ordinal>> {
    private:
      struct Enabler {};

    public:
      static_assert(btas::is_index<_Index>::value, "RangeNd<_Index> instantiated with an _Index type that does not meet the TWG.Index concept");
      typedef RangeNd this_type;
      typedef _Index index_type; ///< index type
      typedef typename _Index::value_type index1_type; ///< 1-index type
      const static blas::Layout order = _Order;

      typedef typename _Ordinal::value_type ordinal_type; ///< Ordinal value type

      // ordinal iterator
      // to be efficient, implemented as iterator that updates index and ordinal at the same time
      typedef std::pair<index_type, ordinal_type> subiter_value_type;
      typedef RangeIterator<subiter_value_type, RangeNd> ordinal_subiterator;
      typedef ::boost::transform_iterator< btas::second_of_pair<subiter_value_type>,
                                           ordinal_subiterator > ordinal_iterator; ///< Ordinal iterator
      typedef ordinal_iterator const_ordinal_iterator; ///< Ordinal interator = Ordinal const_iterator

      typedef BaseRangeNd< RangeNd<_Order, _Index, _Ordinal> > base_type; ///< Parent type
      friend class BaseRangeNd< RangeNd<_Order, _Index, _Ordinal> >;
      template <blas::Layout _O,
                typename _I,
                typename _Ord>
      friend class RangeNd;

      typedef typename base_type::extent_type extent_type;

      /// Default constructor

      /// Construct a range with size and dimensions equal to zero.
      RangeNd() :
        base_type(), ordinal_()
      { }

      /// Constructor defined by the upper and lower bounds

      /// \tparam Index1 any type for which \c btas::is_index<Index1>::value is true
      /// \tparam Index2 any type for which \c btas::is_index<Index2>::value is true
      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      template <typename Index1, typename Index2>
      RangeNd(const Index1& lobound, const Index2& upbound,
              typename std::enable_if<btas::is_index<Index1>::value && btas::is_index<Index2>::value, Enabler>::type = Enabler()) :
        base_type(lobound, upbound), ordinal_(lobound, upbound)
      {
      }

      /// "Move" constructor defined by the upper and lower bounds

      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      RangeNd(index_type&& lobound, index_type&& upbound) :
        base_type(lobound, upbound), ordinal_(lobound, upbound)
      {
      }

      /// Constructor defined by the upper and lower bounds, and the axes strides

      /// \tparam Index1 any type for which \c btas::is_index<Index1>::value is true
      /// \tparam Index2 any type for which \c btas::is_index<Index2>::value is true
      /// \tparam Extent any type for which \c Ordinal(Index1,Index2,Extent) is a valid expression (similar to \c extent_type)
      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      /// \param stride The axes strides of the N-dimensional range
      template <typename Index1, typename Index2, typename Extent>
      RangeNd(const Index1& lobound, const Index2& upbound, const Extent& stride,
              typename std::enable_if<btas::is_index<Index1>::value &&
                                      btas::is_index<Index2>::value &&
                                      btas::is_index<Extent>::value, Enabler>::type = Enabler()) :
        base_type(lobound, upbound), ordinal_(lobound, upbound, stride)
      {
      }

      /// "Move" constructor defined by the upper and lower bounds, and the axes strides

      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      /// \param stride The axes strides of the N-dimensional range
      RangeNd(index_type&& lobound, index_type&& upbound, extent_type&& stride) :
        base_type(lobound, upbound), ordinal_(lobound, upbound, stride)
      {
      }

      /// Constructor defined by the upper and lower bounds, and the ordinal object

      /// \tparam Index1 any type for which \c btas::is_index<Index1>::value is true
      /// \tparam Index2 any type for which \c btas::is_index<Index2>::value is true
      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      template <typename Index1, typename Index2>
      RangeNd(const Index1& lobound, const Index2& upbound, _Ordinal&& ord,
              typename std::enable_if<btas::is_index<Index1>::value && btas::is_index<Index2>::value, Enabler>::type = Enabler()) :
        base_type(lobound, upbound), ordinal_(ord)
      {
      }


      /// "Move" constructor defined by the upper and lower bounds, and the ordinal object

      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      /// \param ordinal The ordinal object
      RangeNd(index_type&& lobound, index_type&& upbound, _Ordinal&& ord) :
        base_type(lobound, upbound), ordinal_(ord)
      {
      }

      /// Range constructor from extent

      /// \tparam Extent An array type convertible to \c extent_type
      /// \param extent An array with the extent of each dimension
      template <typename Extent,
                class = typename std::enable_if<btas::is_index<Extent>::value>::type>
      RangeNd(const Extent& extent) :
        base_type()
      {
          index_type lb = array_adaptor<index_type>::construct(extent.size(), 0);
          base_type::init(lb, extent);
          ordinal_ = _Ordinal(lb, extent);
      }

      /// Range constructor from a pack of extents for each dimension

      /// \tparam _extent0 An integer
      /// \tparam _extents A pack of integers
      /// \param extent0 The extent of first dimension (0)
      /// \param extents A pack of sizes for dimensions 1+
      template<typename _extent0, typename... _extents, class = typename std::enable_if<std::is_integral<_extent0>::value>::type>
      explicit RangeNd(const _extent0& extent0, const _extents&... extents) :
        base_type()
      {
        typedef typename std::common_type<_extent0, typename extent_type::value_type>::type common_type;
        // make initializer_list
        auto range_extent = {static_cast<common_type>(extent0), static_cast<common_type>(extents)...};
        index_type lb = array_adaptor<index_type>::construct(range_extent.size(), 0);
        base_type::init(lb, range_extent);
        ordinal_ = _Ordinal(lb, range_extent);
      }

      /// to construct from an initializer list give it as {extent0, extent1, ... extentN}
      template <typename T>
      RangeNd(std::initializer_list<T> extents,
              typename std::enable_if<std::is_integral<T>::value>::type* = 0) :
        base_type()
      {
        index_type lb = array_adaptor<index_type>::construct(extents.size(), 0);
        base_type::init(lb, extents);
        ordinal_ = _Ordinal(lb, extents);
      }

      /// to construct from an initializer list give it as {extent0, extent1, ... extentN}
      template <typename T1, typename T2>
      RangeNd(std::initializer_list<T1> lobound, std::initializer_list<T2> upbound,
              typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value>::type* = 0) :
        base_type()
      {
        assert(lobound.size() == upbound.size());
        base_type::init(lobound, upbound);
        ordinal_ = _Ordinal(lobound, upbound);
      }

      /// to construct from an initializer list give it as {Range1d_0, Range1d_1, ... Range1d_N}
      template <typename T>
      RangeNd(std::initializer_list<Range1d<T>> range1s) :
        base_type()
      {
        for(auto i: range1s)
          assert(i.stride() == 1);

        std::vector<long> lb(range1s.size());
        std::vector<long> ub(range1s.size());
        int c=0;
        for(auto i: range1s) {
          lb[c] = i.lobound();
          ub[c] = i.upbound();
          ++c;
        }

        base_type::init(lb, ub);
        ordinal_ = _Ordinal(lb, ub);
      }

      /// to construct RangeNd from Range1d given N {Range1d, Range1d, ... Range1d}
      template <typename T>
      RangeNd(Range1d<T> range1, size_type n) :
        base_type()
      {
        assert(range1.stride() == 1);

        std::vector<long> lb(n, range1.lobound());
        std::vector<long> ub(n, range1.upbound());

        base_type::init(lb, ub);
        ordinal_ = _Ordinal(lb, ub);
      }

      /// Copy Constructor

      /// \param other The range to be copied
      RangeNd(const RangeNd& other) :
        base_type(static_cast<base_type>(other)),
        ordinal_(other.ordinal_)
      {
      }

      /// copy constructor from another instantiation of Range
      template <blas::Layout _O,
                typename _I,
                typename _Ord>
      RangeNd (const RangeNd<_O,_I,_Ord>& x) :
        base_type(),
        ordinal_(x.ordinal_)
      {
        base_type::init(x.lobound(), x.upbound());
      }

      /// Move Constructor

      /// \param other The range to be moved
      RangeNd(RangeNd&& other) :
        base_type(other),
        ordinal_(other.ordinal_)
      {
      }

      /// Destructor
      ~RangeNd() { }

      /// Copy assignment operator

      /// \param other The range to be copied
      /// \return A reference to this object
      /// \throw std::bad_alloc When memory allocation fails.
      RangeNd& operator=(const RangeNd& other) {
        this->base_type::operator=(static_cast<const base_type&>(other));
        ordinal_ = other.ordinal_;

        return *this;
      }

      /// Move assignment operator

      /// \param other The range to be moved
      /// \return A reference to this object
      /// \throw std::bad_alloc When memory allocation fails.
      RangeNd& operator=(RangeNd&& other) {
        this->base_type::operator=(static_cast<base_type&&>(other));
        ordinal_ = other.ordinal_;

        return *this;
      }

      /// \return a const reference to the ordinal object
      const _Ordinal& ordinal() const {
        return ordinal_;
      }

      /// swaps the contents of \c *this with \c other
      void swap(RangeNd& other) noexcept {
        base_type::swap(other);
        using std::swap;
        swap(ordinal_, other.ordinal_);
      }

      /// calculates the ordinal value of \c i

      /// Convert an index to its ordinal.
      /// \tparam Index A coordinate index type (array type)
      /// \param index The index to be converted to an ordinal index
      /// \return The ordinal index of \c index
      /// \throw When \c index is not included in this range.
      template <typename Index>
      typename std::enable_if<btas::is_index<Index>::value, ordinal_type>::type
      ordinal(const Index& index) const {
        return ordinal_(index);
      }

      /// calculates the ordinal value of \c i

      /// Convert an index to its ordinal.
      /// \tparam Index A coordinate index type (array type)
      /// \param index The index to be converted to an ordinal index
      /// \return The ordinal index of \c index
      /// \throw When \c index is not included in this range.
      template <typename ... Index>
      typename std::enable_if<not btas::is_index<typename std::decay<Index>::type...>::value, ordinal_type>::type
      ordinal(Index&& ... index) const {
        return ordinal_(std::forward<Index>(index)...);
      }

      /// Range stride accessor (just a shortcut to ordinal().stride() )

      /// \return An Index type that contains the stride of each dimension
      /// \throw nothing
      auto stride() const -> decltype(this->ordinal().stride()) {
        return this->ordinal().stride();
      }

      /// Range stride data accessor (just a shortcut to ordinal().stride_data() )

      /// \return A pointer to a sequence of integers that contains the strides of each dimension
      /// \throw nothing
      auto stride_data() const -> decltype(this->ordinal().stride_data()) {
        return this->ordinal().stride_data();
      }

      /// Constructs a Range slice defined by the upper and lower bounds within this Range

      /// \tparam Index1 An array type convertible to \c index_type
      /// \tparam Index2 An array type convertible to \c index_type
      /// \param lobound The lower bound of the new range
      /// \param upbound The upper bound of the new range
      template <typename Index1, typename Index2>
      typename std::enable_if<btas::is_index<Index1>::value && btas::is_index<Index2>::value, RangeNd>::type
      slice(const Index1& lobound, const Index2& upbound) const
      {
        return RangeNd(lobound, upbound, _Ordinal(this->lobound(), this->upbound(), this->ordinal().stride()));
      }

      /// Constructs a Range slice defined by a subrange for each dimension
      template <typename U>
      RangeNd
      slice(std::initializer_list<Range1d<U>> range1s) const
      {
        for(auto i: range1s)
          assert(i.stride() == 1);

        btas::DEFAULT::index<long> lb(range1s.size());
        btas::DEFAULT::index<long> ub(range1s.size());
        int c=0;
        for(auto i: range1s) {
          lb[c] = i.lobound();
          ub[c] = i.upbound();
          ++c;
        }

        return RangeNd(std::move(lb), std::move(ub), _Ordinal(this->lobound(), this->upbound(), this->ordinal().stride()));
      }

      using base_type::includes;

      /// Check the index ordinal to make sure it is within the range.

      /// \tparam IndexOrdinal An integral type
      /// \param indexord The index ordinal to check for inclusion in the range
      /// equal to the size of the index.
      template <typename IndexOrdinal>
      typename std::enable_if<std::is_integral<IndexOrdinal>::value, bool>::type
      includes(const IndexOrdinal& indexord) const {
        return ordinal_.includes(indexord);
      }

      using base_type::increment;
      /// Increments <index,ordinal> pair
      /// \param[in,out] pair<index,ordinal> to be incremented
      void increment(subiter_value_type& i) const {

        for(auto d: dim_range<order>(this->rank())) {
          // increment subindex
          ++i.first[d];

          // break if done
          if(i.first[d] < this->upbound_[d]) {
            i.second += ordinal_.stride()[d];
            return;
          }

          // Reset current subindex to lobound value and move to the next
          i.second -= (this->upbound_[d] - this->lobound_[d] - 1) * ordinal_.stride()[d];
          i.first[d] = this->lobound_[d];
        }

        // if outside the range, point to the upper bound ... Range::end() will evaluate to upbound also! Range will use this
        std::copy(std::begin(this->upbound_), std::end(this->upbound_), std::begin(i.first));
        i.second = ordinal(i.first);
      }

    private:
      /// The Ordinal object
      _Ordinal ordinal_;

    };

    /// Range Traits
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal>
    struct range_traits<RangeNd<_Order, _Index, _Ordinal> > {
        const static blas::Layout order = _Order;
        typedef _Index index_type;
        typedef typename _Ordinal::value_type ordinal_type;
        constexpr static const bool is_general_layout = true;
    };

    using Range = RangeNd<>;

    /// Range output operator

    /// \param os The output stream that will be used to print \c r
    /// \param r The range to be printed
    /// \return A reference to the output stream
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal>
    std::ostream& operator<<(std::ostream& os, const RangeNd<_Order,_Index, _Ordinal>& r) {
      os << "[";
      array_adaptor<_Index>::print(r.lobound(), os);
      os << ",";
      array_adaptor<_Index>::print(r.upbound(), os);
      os << ")_" << (_Order == blas::Layout::RowMajor ? "R" : "C");
      os << ":" << r.ordinal();
      return os;
    }

    /// swaps the contents of \c r0 with \c r1
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal
             >
    inline void swap(RangeNd<_Order,_Index,_Ordinal>& r0, RangeNd<_Order,_Index,_Ordinal>& r1) noexcept {
      r0.swap(r1);
    }

    /// Range equality comparison

    /// \param r1 The first range to be compared
    /// \param r2 The second range to be compared
    /// \return \c true when \c r1 represents the same range as \c r2, otherwise
    /// \c false.
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal
             >
    inline bool operator ==(const RangeNd<_Order,_Index,_Ordinal>& r1, const RangeNd<_Order,_Index,_Ordinal>& r2) {
      return ((r1.lobound() == r2.lobound()) && (r1.extent() == r2.extent()));
    }

    /// Range inequality comparison

    /// \param r1 The first range to be compared
    /// \param r2 The second range to be compared
    /// \return \c true when \c r1 does not represent the same range as \c r2,
    /// otherwise \c false.
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal
             >
    inline bool operator !=(const RangeNd<_Order,_Index,_Ordinal>& r1, const RangeNd<_Order,_Index,_Ordinal>& r2) {
      return ! operator ==(r1, r2);
    }

    /// Tests congruency of two Ranges

    /// Ranges are congruent if the have identical extents. The congruency of \c r1 and \c r2 of rank N is checked by the following code:
    /// \code
    ///   if (_Order1 == _Order2)
    ///     result = r1.extent()[0] == r2.extent()[0] && r1.extent()[1] == r2.extent()[1] && ... ;
    ///   else
    ///     result = r1.extent()[0] == r2.extent()[N-1] && r1.extent()[1] == r2.extent()[N-2] && ... ;
    /// \endcode
    /// \tparam _Order1
    /// \tparam _Index1
    /// \tparam _Ordinal1
    /// \tparam _Order2
    /// \tparam _Index2
    /// \tparam _Ordinal2
    /// \param r1 a RangeNd<_Order1,_Index1,_Ordinal1> object
    /// \param r2 a RangeNd<_Order2,_Index2,_Ordinal2> object
    /// \return \c true when \c r1 and \c r2 have same extents, otherwise \c false
    /// \note To compare also lobound (except when the ranges have diffferent Order) use Range::operator==()
    template <blas::Layout _Order1,
              typename _Index1,
              typename _Ordinal1,
              blas::Layout _Order2,
              typename _Index2,
              typename _Ordinal2
             >
    inline bool congruent(const RangeNd<_Order1,_Index1,_Ordinal1>& r1,
                          const RangeNd<_Order2,_Index2,_Ordinal2>& r2) {
      const auto r1_extent = r1.extent();
      auto r2_extent = r2.extent(); // no std::crbegin even in C++14, hence no const here
      if (_Order1 == _Order2)
        // 7/15/2014: broken with clang++/libc++ (clang-503.0.40) on OS X
        //auto eq =  std::equal(std::cbegin(r1.extent()), std::cend(r1.extent()),
        //                     std::cbegin(r2.extent()));
        return std::equal(std::cbegin(r1_extent), std::cend(r1_extent),
                          std::cbegin(r2_extent));
      else
        return std::equal(std::cbegin(r1_extent), std::cend(r1_extent),
                          std::rbegin(r2_extent));
    }

    /// Tests whether a range is contiguous, i.e. whether its ordinal values form a contiguous range

    /// \param range a Range
    /// \return true if \p range is contiguous
    template <blas::Layout _Order,
        typename _Index,
        typename _Ordinal>
    inline bool is_contiguous(const RangeNd<_Order, _Index, _Ordinal>& range) {
      return range.ordinal().contiguous();
    }

    /// Permutes a Range

    /// permutes the dimensions using permutation \c p = {p[0], p[1], ... }; for example, if \c lobound() initially returned
    /// {lb[0], lb[1], ... }, after this call \c lobound() will return {lb[p[0]], lb[p[1]], ...}.
    /// \param perm an array specifying permutation of the dimensions
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal,
              typename AxisPermutation,
              class = typename std::enable_if<btas::is_index<AxisPermutation>::value>::type>
    RangeNd<_Order, _Index>
    permute(const RangeNd<_Order, _Index, _Ordinal>& r,
            const AxisPermutation& perm)
    {
      const auto rank = r.rank();
      auto lb = r.lobound();
      auto ub = r.upbound();

      typedef typename RangeNd<_Order, _Index, _Ordinal>::index_type index_type;
      index_type lobound, upbound;
      lobound = array_adaptor<index_type>::construct(rank);
      upbound = array_adaptor<index_type>::construct(rank);

      std::for_each(std::begin(perm), std::end(perm), [&](const typename AxisPermutation::value_type& i){
        const auto pi = *(std::begin(perm) + i);
        *(std::begin(lobound)+i) = *(std::begin(lb) + pi);
        *(std::begin(upbound)+i) = *(std::begin(ub) + pi);
      });

      return RangeNd<_Order, _Index, _Ordinal>(std::move(lobound), std::move(upbound), permute(r.ordinal(), perm) );
    }

    /// Permutes a Range

    /// permutes the axes using permutation \c p = {p[0], p[1], ... }; for example, if \c lobound() initially returned
    /// {lb[0], lb[1], ... }, after this call \c lobound() will return {lb[p[0]], lb[p[1]], ...} .
    /// \param perm an array specifying permutation of the axes
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal,
              typename T>
    RangeNd<_Order, _Index, _Ordinal>
    permute(const RangeNd<_Order, _Index, _Ordinal>& r,
            std::initializer_list<T> perm)
    {
      typename RangeNd<_Order, _Index, _Ordinal>::extent_type p = array_adaptor<typename RangeNd<_Order, _Index, _Ordinal>::extent_type>::construct(perm.size());
      std::copy(std::begin(perm), std::end(perm), std::begin(p));
      return permute(r, p);
    }

    /// Takes the diagonal part of a range

    /// Given a RangeNd, returns a new RangeNd whose indices increase in lock step.
    /// Requires \c lobound() to be uniform {n,n,n,...}.
    /// Iterating over the returned range yields:
    /// {n,n,n,...}
    /// {n+1,n+1,n+1,...}
    /// {n+2,n+2,n+2,...}
    /// up to \c upbound()
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal>
    RangeNd<_Order, _Index>
    diag(const RangeNd<_Order, _Index, _Ordinal>& r)
      {
      if(r.rank() == 0ul) return r;
      using index_value = typename RangeNd<_Order,_Index>::index_type::value_type;
      index_value stride = 1,
                  prod_extents = 1,
                  extent = r.upbound()[0];
      const auto dr = _Order == blas::Layout::RowMajor ? Range1(r.rank()-1,0,-1)
                                              : Range1(0,r.rank()-1,1);
      for(const auto i : dr)
        {
        assert(r.lobound()[0] == r.lobound()[i]);
        prod_extents *= (r.upbound()[i]-r.lobound()[i]);
        stride += prod_extents;
        extent = std::min(extent,r.upbound()[i]);
        }
      return RangeNd<_Order,_Index>({r.lobound()[0]},{extent},{static_cast<typename RangeNd<_Order, _Index>::extent_type::value_type>(stride)});
      }

    /// Group a set of adjacent indices of a Range

    /// Combine/group/flatten a set of adjacent indices into a single index.
    /// Groups the indices from [istart,iend) not including iend.
    /// If the original indices have extents e1,e2,e3,... the grouped index
    /// will have extent e1*e2*e3*...
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal>
    RangeNd<_Order, _Index,_Ordinal>
    group(const RangeNd<_Order, _Index, _Ordinal>& r,
          size_t istart,
          size_t iend)
      {
      using index_type = typename RangeNd<_Order,_Index,_Ordinal>::index_type;

      if(r.rank() == 0 || iend <= (istart+1)) return r;
      const auto ngroup = iend-istart;
      const auto newr = r.rank()-ngroup+1;
      assert(ngroup >= 2);
      assert(r.rank() >= ngroup);
      assert(iend > 0);

      index_type lobound(newr),
                 upbound(newr);
      for(size_t i = 0; i < istart; ++i)
          {
          lobound[i] = r.lobound()[i];
          upbound[i] = r.upbound()[i];
          }
      lobound[istart] = 0;
      upbound[istart] = 1;
      for(size_t i = istart; i < iend; ++i)
          {
          upbound[istart] *= (r.upbound()[i]-r.lobound()[i]);
          }
      for(size_t i = iend, j = istart+1; i < r.rank(); ++i,++j)
          {
          lobound[j] = r.lobound()[i];
          upbound[j] = r.upbound()[i];
          }

      return RangeNd<_Order,_Index,_Ordinal>(lobound,upbound);
      }

    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal>
    RangeNd<_Order, _Index,_Ordinal>
    flatten(const RangeNd<_Order, _Index, _Ordinal>& r)
      {
      using index_value = typename RangeNd<_Order,_Index,_Ordinal>::index_type::value_type;
      index_value lobound = 0,
                  upbound = 1;
      for(size_t i = 0; i < r.rank(); ++i)
          {
          upbound *= (r.upbound()[i]-r.lobound()[i]);
          }
      return RangeNd<_Order,_Index,_Ordinal>({lobound},{upbound});
      }

    ///
    /// Tie (i.e. lock or fuse) N indices together, returning a range with (N-1) fewer indices.
    /// The position of the tied index is the position of the first index in the group.
    /// Example:
    /// std::vector<std::size_t> inds = { 0, 2 };
    /// tie(T,inds)(i,j) = T(i,j,i)
    ///
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal,
              typename ArrayType>
    RangeNd<_Order, _Index,_Ordinal>
    tieIndex(const RangeNd<_Order, _Index, _Ordinal>& r,
             const ArrayType& inds)
      {
      using index_type = typename RangeNd<_Order,_Index,_Ordinal>::index_type;
      using index_value = typename index_type::value_type;

      if(inds.size() < 2) return r;
      assert(inds.size() <= r.rank());

      auto newr = r.rank()-(inds.size()-1);
      auto ti = inds[0];
      auto tbegin = r.lobound()[ti];
      auto tend = r.upbound()[ti];
      for(const auto i : inds)
          {
          assert(i < r.rank());
          ti = std::min(ti,i);
          tbegin = std::max(tbegin,r.lobound()[i]);
          tend = std::min(tend,r.upbound()[i]);
          }
      if(ti >= newr) ti = newr-1;

      index_type lobound(newr),
                 upbound(newr),
                 stride(newr);

      stride[ti] = 0;
      lobound[ti] = tbegin;
      upbound[ti] = tend;

      const auto dr = (_Order == blas::Layout::RowMajor) ? Range1(r.rank()-1,-1,-1)
                                                : Range1(0,r.rank(),1);
      const auto nr = (_Order == blas::Layout::RowMajor) ? Range1(newr-1,-1,-1)
                                                : Range1(0,newr,1);
      index_value prod_extents = 1;
      auto it = nr.begin();
      for(const auto i : dr)
          {
          bool is_tied = false;
          for(auto j : inds) if(i == j)
              {
              is_tied = true;
              break;
              }
          if(is_tied)
              {
              stride[ti] += prod_extents;
              }
          else
              {
              if(*it == ti) ++it;
              stride[*it] = prod_extents;
              lobound[*it] = r.lobound()[i];
              upbound[*it] = r.upbound()[i];
              ++it;
              }
          prod_extents *= (r.upbound()[i]-r.lobound()[i]);
          }

      return RangeNd<_Order,_Index,_Ordinal>(lobound,upbound,stride);
      }

    ///
    /// tieIndex wrapper taking a variadic list of integers
    ///
    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal,
              typename... _args>
    RangeNd<_Order, _Index,_Ordinal>
    tieIndex(const RangeNd<_Order, _Index, _Ordinal>& r,
             size_t i0,
             const _args&... rest)
        {
        const auto size = 1 + sizeof...(rest);
        std::array<size_t,size> inds = { i0, static_cast<size_t>(rest)...};
        return tieIndex(r,inds);
        }


    template <blas::Layout _Order,
              typename _Index,
              typename _Ordinal>
    class boxrange_iteration_order< btas::RangeNd<_Order, _Index, _Ordinal> > {
      public:
        enum {row_major = boxrange_iteration_order<void>::row_major,
              other = boxrange_iteration_order<void>::other,
              column_major = boxrange_iteration_order<void>::column_major};

        static constexpr int value = (_Order == blas::Layout::RowMajor) ? row_major : column_major;
    };
}  // namespace btas

//
//  Default range type
//

namespace btas {
namespace DEFAULT {

using range = btas::Range;

}  // namespace DEFAULT
}  // namespace btas


#ifdef BTAS_HAS_BOOST_SERIALIZATION
namespace boost {
namespace serialization {

  /// boost serialization
  template<class Archive, blas::Layout _Order,
           typename _Index, typename _Ordinal>
  void serialize(Archive& ar, btas::RangeNd<_Order, _Index, _Ordinal>& t, const unsigned int version) {
    boost::serialization::split_free(ar, t, version);
  }
  template<class Archive, blas::Layout _Order,
           typename _Index, typename _Ordinal>
  void save(Archive& ar, const btas::RangeNd<_Order, _Index, _Ordinal>& t, const unsigned int version) {
    auto lobound = t.lobound();
    auto upbound = t.upbound();
    auto ordinal = t.ordinal();
    ar << BOOST_SERIALIZATION_NVP(lobound) << BOOST_SERIALIZATION_NVP(upbound) << BOOST_SERIALIZATION_NVP(ordinal);
  }
  template<class Archive, blas::Layout _Order,
           typename _Index, typename _Ordinal>
  void load(Archive& ar, btas::RangeNd<_Order, _Index, _Ordinal>& t, const unsigned int version) {
    typedef typename btas::BaseRangeNd<btas::RangeNd<_Order, _Index, _Ordinal>>::index_type index_type;
    index_type lobound, upbound;
    _Ordinal ordinal;
    ar >> BOOST_SERIALIZATION_NVP(lobound) >> BOOST_SERIALIZATION_NVP(upbound) >> BOOST_SERIALIZATION_NVP(ordinal);
    t = btas::RangeNd<_Order, _Index, _Ordinal>(std::move(lobound), std::move(upbound), std::move(ordinal));
  }

}
}
#endif

// serialization to/fro MADNESS archive (github.com/m-a-d-n-e-s-s/madness)
namespace madness {
  namespace archive {

    template <class Archive, blas::Layout _Order, typename _Index,
        typename _Ordinal>
    struct ArchiveLoadImpl<Archive, btas::RangeNd<_Order, _Index, _Ordinal>> {
      static inline void load(const Archive& ar,
                              btas::RangeNd<_Order, _Index, _Ordinal>& r) {
        typedef typename btas::BaseRangeNd<
            btas::RangeNd<_Order, _Index, _Ordinal>>::index_type index_type;
        index_type lobound{}, upbound{};
        _Ordinal ordinal{};
        ar& lobound& upbound& ordinal;
        r = btas::RangeNd<_Order, _Index, _Ordinal>(
            std::move(lobound), std::move(upbound), std::move(ordinal));
      }
    };

    template <class Archive, blas::Layout _Order, typename _Index,
        typename _Ordinal>
    struct ArchiveStoreImpl<Archive, btas::RangeNd<_Order, _Index, _Ordinal>> {
      static inline void store(const Archive& ar,
                               const btas::RangeNd<_Order, _Index, _Ordinal>& r) {
        ar& r.lobound() & r.upbound() & r.ordinal();
      }
    };

  }  // namespace archive
}  // namespace madness

#endif /* BTAS_RANGE_H_ */
