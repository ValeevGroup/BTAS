/*
 * ordinal.h
 *
 *  Created on: Dec 26, 2013
 *      Author: evaleev
 */

#ifndef BTAS_ORDINAL_H_
#define BTAS_ORDINAL_H_

#include <cassert>

#include <btas/types.h>
#include <btas/defaults.h>
#include <btas/array_adaptor.h>
#include <btas/index_traits.h>

namespace btas {

  /// BoxOrdinal is an implementation detail of BoxRange.
  /// It maps the index to its ordinal value. It also knows whether
  /// the map is contiguous (i.e. whether adjacent indices have adjacent ordinal
  /// values).
  template <CBLAS_ORDER _Order = CblasRowMajor,
            typename _Index = btas::DEFAULT::index_type,
            class = typename std::enable_if<btas::is_index<_Index>::value>
           >
  class BoxOrdinal {
    public:
      typedef _Index index_type;
      const static CBLAS_ORDER order = _Order;
      typedef int64_t value_type;
      typedef typename btas::replace_value_type<_Index,value_type>::type stride_type;    ///< stride type

      template <CBLAS_ORDER _O,
                typename _I,
                class _X
               >
      friend class BoxOrdinal;

      BoxOrdinal() {
        assert((contiguous_ = false) || true); // workaround for Boost serialization
                                               // it breaks Debug builds when reading uninitialized bools
      }

      template <typename Index1,
                typename Index2,
                class = typename std::enable_if<btas::is_index<typename std::decay<Index1>::type>::value && btas::is_index<typename std::decay<Index2>::type>::value>::type
               >
      BoxOrdinal(Index1&& lobound,
                 Index2&& upbound) {
          init(std::forward<Index1>(lobound), std::forward<Index2>(upbound));
      }

      template <typename Index1,
                typename Index2,
                typename Stride,
                class = typename std::enable_if<btas::is_index<typename std::decay<Index1>::type>::value &&
                                                btas::is_index<typename std::decay<Index2>::type>::value &&
                                                btas::is_index<typename std::decay<Stride>::type>::value>::type
               >
      BoxOrdinal(Index1&& lobound,
                 Index2&& upbound,
                 Stride&& stride) {
          init(std::forward<Index1>(lobound), std::forward<Index2>(upbound), std::forward<Stride>(stride));
      }

      BoxOrdinal(stride_type&& stride,
                 value_type&& offset,
                 bool cont) : stride_(stride), offset_(offset), contiguous_(cont) {
      }


      BoxOrdinal(const BoxOrdinal& other) :
        stride_(other.stride_),
        offset_(other.offset_),
        contiguous_ (other.contiguous_) {
      }

      template <CBLAS_ORDER _O,
                typename _I,
                class = typename std::enable_if<btas::is_index<_I>::value>
               >
      BoxOrdinal(const BoxOrdinal<_O,_I>& other) {
          auto n = other.rank();
          stride_ = array_adaptor<stride_type>::construct(n);
          using std::cbegin;
          using std::begin;
          using std::cend;
          std::copy(cbegin(other.stride_), cend(other.stride_),
                    begin(stride_));
          offset_ = other.offset_;
          contiguous_ = other.contiguous_;
      }

      ~BoxOrdinal() {}

      std::size_t rank() const {
        using btas::rank;
        return rank(stride_);
      }

      const stride_type& stride() const {
        return stride_;
      }

      // no easy way without C++14 to invoke data(stride) in ADL-capable way
#if __cplusplus < 201402L
      auto stride_data() const -> decltype(std::data(this->stride())) {
        return std::data(stride_);
      }
#else
      auto stride_data() const {
        using std::data;
        return data(stride_);
      }
#endif
      value_type offset() const {
        return offset_;
      }

      bool contiguous() const {
        return contiguous_;
      }

      template <typename Index>
      typename std::enable_if<btas::is_index<Index>::value, value_type>::type
      operator()(const Index& index) const {
        assert(index.size() == rank());
        value_type o = 0;
        const auto end = this->rank();
        using std::cbegin;
        for(std::size_t i = 0; i != end; ++i)
          o += *(cbegin(index) + i) * *(cbegin(this->stride_) + i);

        return o - offset_;
      }

      /// computes the ordinal value using a pack of indices
      template <typename ... Index>
      typename std::enable_if<not btas::is_index<typename std::decay<Index>::type...>::value, value_type>::type
      operator()(Index&& ... index) const {
        assert(sizeof...(index) == rank());
        using std::cbegin;
        value_type o = zip(cbegin(this->stride_), std::forward<Index>(index)...);
        return o - offset_;
      }

    private:
      template <typename Iterator, typename FirstIndex, typename ... RestOfIndices>
      value_type zip(Iterator&& it, FirstIndex&& index, RestOfIndices&& ... rest) const {
        return *it * index + zip(it+1, rest...);
      }
      template <typename Iterator>
      value_type zip(Iterator&& it) const {
        return 0;
      }

    public:

      /// Does ordinal value belong to this ordinal range?
      template <typename I>
      typename std::enable_if<std::is_integral<I>::value, bool>::type
      includes(const I& ord) const {
        assert(false && "BoxOrdinal::includes() is not not yet implemented");
      }

    private:

      template <typename Index1,
                typename Index2
               >
      void init(const Index1& lobound,
                const Index2& upbound) {
        using btas::rank;
        auto n = rank(lobound);
        if (n == 0) return;

        value_type volume = 1;
        offset_ = 0;
        stride_ = array_adaptor<stride_type>::construct(n);

        // Compute range data
        if (order == CblasRowMajor) {
          for(typename std::make_signed<decltype(n)>::type i = n - 1;
              i >= 0; --i) {
            stride_[i] = volume;
            using std::cbegin;
            auto li = *(cbegin(lobound) + i);
            auto ui = *(cbegin(upbound) + i);
            offset_ += li * volume;
            volume *= (ui - li);
          }
        }
        else {
          for(decltype(n) i = 0; i != n; ++i) {
            stride_[i] = volume;
            using std::cbegin;
            auto li = *(cbegin(lobound) + i);
            auto ui = *(cbegin(upbound) + i);
            offset_ += li * volume;
            volume *= (ui - li);
          }
        }
        contiguous_ = true;
      }

      /// upbound only needed to check contiguousness
      template <typename Index1,
                typename Index2,
                typename Weight
               >
      void init(const Index1& lobound,
                const Index2& upbound,
                const Weight& stride) {
        using btas::rank;
        auto n = rank(lobound);
        if (n == 0) return;

        value_type volume = 1;
        offset_ = 0;
        stride_ = array_adaptor<stride_type>::construct(n);
        using std::cbegin;
        using std::begin;
        using std::cend;
        std::copy(cbegin(stride), cend(stride), begin(stride_));

        // Compute offset and check whether contiguous
        contiguous_ = true;
        if (order == CblasRowMajor) {
          for(typename std::make_signed<decltype(n)>::type i = n - 1;
              i >= 0; --i) {
            contiguous_ &= (volume == stride_[i]);
            auto li = *(cbegin(lobound) + i);
            auto ui = *(cbegin(upbound) + i);
            offset_ += li * stride_[i];
            volume *= (ui - li);
          }
        }
        else {
          for(decltype(n) i = 0; i != n; ++i) {
            contiguous_ &= (volume == stride_[i]);
            auto li = *(cbegin(lobound) + i);
            auto ui = *(cbegin(upbound) + i);
            offset_ += li * stride_[i];
            volume *= (ui - li);
          }
        }
      }

#ifdef BTAS_HAS_BOOST_SERIALIZATION
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive& ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(stride_) & BOOST_SERIALIZATION_NVP(offset_) & BOOST_SERIALIZATION_NVP(contiguous_);
      }
#endif

      stride_type stride_; //!< stride of each dimension (stride in the language of NumPy)
      value_type offset_; //!< lobound . stride so that easy to compute ordinal: ordinal(index) = index . stride - offset
      bool contiguous_; //!< whether index iterator traverses a contiguous sequence of ordinals
  };

  /// Permutes BoxOrdinal

  /// permutes the dimensions using permutation \c p = {p[0], p[1], ... }; for example, if \c stride() initially returned
  /// {s[0], s[1], ... }, after this call \c stride() will return {s[p[0]], s[p[1]], ...}.
  /// \param perm an array specifying permutation of the dimensions
  template <CBLAS_ORDER _Order,
            typename _Index,
            typename AxisPermutation,
            class = typename std::enable_if<btas::is_index<AxisPermutation>::value>::type>
  BoxOrdinal<_Order, _Index> permute(const BoxOrdinal<_Order, _Index>& ord,
                                     const AxisPermutation& perm)
  {
    const auto rank = ord.rank();
    auto st = ord.stride();

    typedef typename BoxOrdinal<_Order, _Index>::stride_type stride_type;
    stride_type stride;
    stride = array_adaptor<stride_type>::construct(rank);

    using std::cbegin;
    using std::begin;
    using std::cend;
    std::for_each(cbegin(perm), cend(perm), [&](const typename AxisPermutation::value_type& i){
      const auto pi = *(cbegin(perm) + i);
      *(begin(stride)+i) = *(cbegin(st) + pi);
    });

    return BoxOrdinal<_Order, _Index>(std::move(stride), ord.offset(), ord.contiguous());
  }

  /// Range output operator

  /// \param os The output stream that will be used to print \c r
  /// \param r The range to be printed
  /// \return A reference to the output stream
  template <CBLAS_ORDER _Order,
            typename _Index>
  inline std::ostream& operator<<(std::ostream& os, const BoxOrdinal<_Order,_Index>& ord) {
    array_adaptor<typename BoxOrdinal<_Order,_Index>::stride_type>::print(ord.stride(), os);
    return os;
  }

} // namespace btas

#endif /* BTAS_ORDINAL_H_ */
