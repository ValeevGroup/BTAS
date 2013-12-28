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
#include <btas/ordinal.h>

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

    /// BaseRangeNd is a <a href="http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern">CRTP</a>
    /// base for implementations of N-dimensional Ranges.

    /**
     * BaseRangeNd defines a box in the index space, and the iteration order on it.
     * The iteration order depends on the CBLAS_ORDER parameter (ordering of dimensions).
     *
     * \tparam _Derived implementation of Range, to be derived from \c BaseRangeNd as \c public \c BaseRangeNd<Derived>
     */
    template <typename _Derived>
    class BaseRangeNd {
    protected:
      const static CBLAS_ORDER order = range_traits<_Derived>::order;
      typedef typename range_traits<_Derived>::index_type index_type; ///< index type
      typedef typename std::make_unsigned<index_type>::type extent_type;    ///< Range extent type
      typedef std::size_t size_type; ///< Size type

      typedef index_type value_type; ///< Range can be viewed as a Container of value_type
      typedef index_type& reference_type;
      typedef const value_type& const_reference_type;
      typedef RangeIterator<index_type, _Derived> const_iterator; ///< Index iterator
      typedef const_iterator iterator; ///< interator = const_iterator
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
          return;
        }
        validate(lobound, upbound);

        lobound_ = array_adaptor<index_type>::construct(n);
        std::copy(lobound.begin(), lobound.end(), lobound_.begin());
        upbound_ = array_adaptor<index_type>::construct(n);
        std::copy(upbound.begin(), upbound.end(), upbound_.begin());
      }

      template <typename Index1, typename Index2>
      void validate(const Index1& lobound, const Index2& upbound) {
        using btas::rank;
        auto n = rank(lobound);
        assert(n == rank(upbound));

        typedef typename common_signed_type<typename Index1::value_type, typename Index2::value_type>::type ctype;
        for(auto i = 0; i != n; ++i) {
          auto li = *(lobound.begin() + i);
          auto ui = *(upbound.begin() + i);
          assert(static_cast<ctype>(li) <= static_cast<ctype>(ui));
        }
      }

    protected:

      /// Default constructor

      /// Construct an unitialized range; its area is zero.
      BaseRangeNd() : lobound_(), upbound_() { }

      /// Constructor defined by the upper and lower bounds

      /// \tparam Index1 An array type convertible to \c index_type
      /// \tparam Index2 An array type convertible to \c index_type
      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      template <typename Index1, typename Index2>
      BaseRangeNd(const Index1& lobound, const Index2& upbound,
              typename std::enable_if<btas::is_index<Index1>::value && btas::is_index<Index2>::value, Enabler>::type = Enabler())
      {
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
        lobound_(other.lobound_), upbound_(other.upbound_)
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
        lobound_(other.lobound_), upbound_(other.upbound_)
      {
      }


      /// Destructor
      ~BaseRangeNd() { }

      /// Copy assignment operator

      /// \param other The range to be copied
      /// \return A reference to this object
      /// \throw std::bad_alloc When memory allocation fails.
      BaseRangeNd& operator=(const BaseRangeNd& other) {
        lobound_ = other.lobound_;
        upbound_ = other.upbound_;

        return *this;
      }

      /// Move assignment operator

      /// \param other The range to be moved
      /// \return A reference to this object
      BaseRangeNd& operator=(BaseRangeNd&& other) {
        lobound_ = other.lobound_;
        upbound_ = other.upbound_;

        return *this;
      }

      void swap(BaseRangeNd& other) {
        std::swap(lobound_, other.lobound_);
        std::swap(upbound_, other.upbound_);
      }

    public:

      /// Access a particular subrange of Range

      /// returns the Range1 corresponding to the dimension \c d
      /// \param d the dimension index
      Range1d<typename index_type::value_type> range(size_t d) const {
        return Range1d<typename index_type::value_type>(*(lobound_.begin()+d), *(upbound_.begin()+d));
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
      const_iterator begin() const { return const_iterator(lobound_, static_cast<const _Derived*>(this)); }

      /// Index iterator factory

      /// The iterator dereferences to an index. The order of iteration matches
      /// the data layout of a dense tensor.
      /// \return An iterator that holds the upbound element index of a tensor
      /// \throw nothing
      const_iterator end() const { return const_iterator(upbound_, static_cast<const _Derived*>(this)); }

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


    private:

      /// Increment the coordinate index \c i in this range

      /// \param[in,out] i The coordinate index to be incremented
      void increment(index_type& i) const {

        if (order == CblasRowMajor) {
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

    private:
      index_type lobound_; ///< range origin
      index_type upbound_; ///< range extent

    }; // class BaseRangeNd

    template <CBLAS_ORDER _Order = CblasRowMajor,
              typename _Index = btas::varray<long>,
              typename _Ordinal = btas::BoxOrdinal<_Order,_Index>,
              class = typename std::enable_if<btas::is_index<_Index>::value>
             >
    class RangeNd : public BaseRangeNd< RangeNd<_Order,_Index> > {
    private:
      struct Enabler {};

    public:
      typedef RangeNd this_type;
      typedef _Index index_type; ///< index type
      const static CBLAS_ORDER order = _Order;

      typedef typename _Ordinal::value_type ordinal_type; ///< Ordinal value type

      typedef index_type value_type; ///< Range can be viewed as a Container of value_type
      typedef index_type& reference_type;
      typedef const value_type& const_reference_type;
      typedef RangeIterator<index_type, RangeNd> const_iterator; ///< Index iterator
      typedef const_iterator iterator; ///< interator = const_iterator

      typedef BaseRangeNd< RangeNd<_Order, _Index, _Ordinal> > base_type; ///< Parent type
      friend class BaseRangeNd< RangeNd<_Order, _Index, _Ordinal> >;
      template <CBLAS_ORDER _O,
                typename _I,
                typename _Ord,
                typename X>
      friend class RangeNd;

      typedef typename base_type::extent_type extent_type;

      /// Default constructor

      /// Construct a range with size and dimensions equal to zero.
      RangeNd() :
        base_type(), ordinal_()
      { }

      /// Constructor defined by the upper and lower bounds

      /// \tparam Index1 An array type convertible to \c index_type
      /// \tparam Index2 An array type convertible to \c index_type
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
        base_type(lobound, upbound), ordinal_(lobound, upbound, weight)
      {
      }

      /// "Move" constructor defined by the upper and lower bounds, and the axes weights

      /// \param lobound The lower bound of the N-dimensional range
      /// \param upbound The upper bound of the N-dimensional range
      /// \param weight The axes weights of the N-dimensional range
      RangeNd(index_type&& lobound, index_type&& upbound, extent_type&& weight) :
        base_type(lobound, upbound), ordinal_(lobound, upbound, weight)
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
      /// \param sizes A pack of sizes for dimensions 1+
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
      RangeNd(std::initializer_list<T> extents) :
        base_type()
      {
        index_type lb = array_adaptor<index_type>::construct(extents.size(), 0);
        base_type::init(lb, extents);
        ordinal_ = _Ordinal(lb, extents);
      }

      /// to construct from an initializer list give it as {extent0, extent1, ... extentN}
      template <typename T1, typename T2>
      RangeNd(std::initializer_list<T1> lobound, std::initializer_list<T2> upbound) :
        base_type()
      {
        assert(lobound.size() == upbound.size());
        base_type::init(lobound, upbound);
        ordinal_ = _Ordinal(lobound, upbound);
      }

      /// Copy Constructor

      /// \param other The range to be copied
      RangeNd(const RangeNd& other) :
        base_type(static_cast<base_type>(other)),
        ordinal_(other.ordinal_)
      {
      }

      /// copy constructor from another instantiation of Range
      template <CBLAS_ORDER _O,
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

      /// returns the ordinal object
      const _Ordinal& ordinal() const {
        return ordinal_;
      }

      /// calculate the ordinal value of \c i

      /// Convert an index to its ordinal.
      /// \tparam Index A coordinate index type (array type)
      /// \param index The index to be converted to an ordinal index
      /// \return The ordinal index of \c index
      /// \throw When \c index is not included in this range.
      template <typename Index>
      typename std::enable_if<btas::is_index<Index>::value, ordinal_type>::type
      ordinal(const Index& index) const {
        this->validate_index(index);
        return ordinal_(index);
      }

    private:

      /// calculate the ordinal index of \c i

      /// Convert an index to its ordinal.
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

      /// The Ordinal object
      _Ordinal ordinal_;

    };

    /// Range Traits
    template <CBLAS_ORDER _Order,
              typename _Index,
              typename _Ordinal>
    struct range_traits<RangeNd<_Order, _Index, _Ordinal> > {
        const static CBLAS_ORDER order = _Order;
        typedef _Index index_type;
    };

    using Range = RangeNd<>;

    /// Range output operator

    /// \param os The output stream that will be used to print \c r
    /// \param r The range to be printed
    /// \return A reference to the output stream
    template <CBLAS_ORDER _Order,
              typename _Index,
              typename _Ordinal,
              typename _X>
    inline std::ostream& operator<<(std::ostream& os, const RangeNd<_Order,_Index, _Ordinal, _X>& r) {
      os << "[";
      array_adaptor<_Index>::print(r.lobound(), os);
      os << ",";
      array_adaptor<_Index>::print(r.upbound(), os);
      os << ")_" << (_Order == CblasRowMajor ? "R" : "C");
      os << ":" << r.ordinal();
      return os;
    }

    /// Exchange the values of the give two ranges.
    template <CBLAS_ORDER _Order,
              typename _Index,
              typename _Ordinal
             >
    inline void swap(RangeNd<_Order,_Index,_Ordinal>& r0, RangeNd<_Order,_Index,_Ordinal>& r1) { // no throw
      r0.swap(r1);
    }


    /// Range equality comparison

    /// \param r1 The first range to be compared
    /// \param r2 The second range to be compared
    /// \return \c true when \c r1 represents the same range as \c r2, otherwise
    /// \c false.
    template <CBLAS_ORDER _Order,
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
    template <CBLAS_ORDER _Order,
              typename _Index,
              typename _Ordinal
             >
    inline bool operator !=(const RangeNd<_Order,_Index,_Ordinal>& r1, const RangeNd<_Order,_Index,_Ordinal>& r2) {
      return ! operator ==(r1, r2);
    }

    /// Permutes a Range

    /// permutes the dimensions using permutation \c p = {p[0], p[1], ... }; for example, if \c lobound() initially returned
    /// {lb[0], lb[1], ... }, after this call \c lobound() will return {lb[p[0]], lb[p[1]], ...}.
    /// \param perm an array specifying permutation of the dimensions
    template <CBLAS_ORDER _Order,
              typename _Index,
              typename _Ordinal,
              typename AxisPermutation,
              class = typename std::enable_if<btas::is_index<AxisPermutation>::value>::type>
    RangeNd<_Order, _Index> permute(const RangeNd<_Order, _Index, _Ordinal>& r,
                                    const AxisPermutation& perm)
    {
      const auto rank = r.rank();
      auto lb = r.lobound();
      auto ub = r.upbound();
      auto wt = r.ordinal().weight();

      typedef typename RangeNd<_Order, _Index, _Ordinal>::index_type index_type;
      typedef typename RangeNd<_Order, _Index, _Ordinal>::extent_type extent_type;
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

      return RangeNd<_Order, _Index, _Ordinal>(std::move(lobound), std::move(upbound), std::move(weight));
    }

    /// Permutes a Range

    /// permutes the axes using permutation \c p = {p[0], p[1], ... }; for example, if \c lobound() initially returned
    /// {lb[0], lb[1], ... }, after this call \c lobound() will return {lb[p[0]], lb[p[1]], ...} .
    /// \param perm an array specifying permutation of the axes
    template <CBLAS_ORDER _Order,
              typename _Index,
              typename _Ordinal,
              typename T>
    RangeNd<_Order, _Index, _Ordinal> permute(const RangeNd<_Order, _Index, _Ordinal>& r,
                                              std::initializer_list<T> perm)
    {
      typename RangeNd<_Order, _Index, _Ordinal>::extent_type p = array_adaptor<decltype(p)>::construct(perm.size());
      std::copy(perm.begin(), perm.end(), p.begin());
      return permute(r, p);
    }


    template <CBLAS_ORDER _Order,
              typename _Index,
              typename _Ordinal>
    class boxrange_iteration_order< btas::RangeNd<_Order, _Index, _Ordinal> > {
      public:
        enum {row_major = boxrange_iteration_order<void>::row_major,
              other = boxrange_iteration_order<void>::other,
              column_major = boxrange_iteration_order<void>::column_major};

        static constexpr int value = (_Order == CblasRowMajor) ? row_major : column_major;
    };
}


namespace boost {
namespace serialization {

  /// boost serialization
  template<class Archive, CBLAS_ORDER _Order,
           typename _Index, typename _Ordinal>
  void serialize(Archive& ar, btas::RangeNd<_Order, _Index, _Ordinal>& t,
                 const unsigned int version) {
    ar & t.lobound() & t.upbound() & t.ordinal();
  }

}
}

#endif /* BTAS_RANGE_H_ */
