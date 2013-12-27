/*
 * ordinal.h
 *
 *  Created on: Dec 26, 2013
 *      Author: evaleev
 */

#ifndef BTAS_ORDINAL_H_
#define BTAS_ORDINAL_H_

#include <btas/types.h>
#include <btas/array_adaptor.h>
#include <btas/index_traits.h>
#include <btas/varray/varray.h>

namespace btas {

  /// BoxOrdinal is an implementation detail of BoxRange.
  /// It maps the index to its ordinal value. It also knows whether
  /// the map is dense (i.e. whether adjacent indices have adjacent ordinal
  /// values).
  template <CBLAS_ORDER _Order = CblasRowMajor,
            typename _Index = btas::varray<long>,
            class = typename std::enable_if<btas::is_index<_Index>::value>
           >
  class BoxOrdinal {
    public:
      typedef _Index index_type;
      const static CBLAS_ORDER order = _Order;
      typedef int64_t value_type;
      typedef typename btas::replace_value_type<_Index,value_type>::type weight_type;    ///< weight type

      template <CBLAS_ORDER _O,
                typename _I,
                class _X
               >
      friend class BoxOrdinal;

      BoxOrdinal() {}

      template <typename Index1,
                typename Index2,
                class = typename std::enable_if<btas::is_index<Index1>::value && btas::is_index<Index2>::value>::type
               >
      BoxOrdinal(const Index1& lobound,
                 const Index2& upbound) {
          init(lobound, upbound);
      }

      template <typename Index1,
                typename Index2,
                typename Weight,
                class = typename std::enable_if<btas::is_index<Index1>::value &&
                                                btas::is_index<Index2>::value &&
                                                btas::is_index<Weight>::value>::type
               >
      BoxOrdinal(const Index1& lobound,
                 const Index2& upbound,
                 const Weight& weight) {
          init(lobound, upbound, weight);
      }


      BoxOrdinal(const BoxOrdinal& other) :
        weight_(other.weight_),
        offset_(other.offset_),
        dense_ (other.dense_) {
      }

      template <CBLAS_ORDER _O,
                typename _I,
                class = typename std::enable_if<btas::is_index<_I>::value>
               >
      BoxOrdinal(const BoxOrdinal<_O,_I>& other) {
          auto n = other.rank();
          weight_ = array_adaptor<weight_type>::construct(n);
          std::copy(other.weight_.begin(), other.weight_.end(),
                    weight_.begin());
          offset_ = other.offset_;
          dense_ = other.dense_;
      }

      ~BoxOrdinal() {}

      std::size_t rank() const {
        using btas::rank;
        return rank(weight_);
      }

      const weight_type& weight() const {
        return weight_;
      }

      bool dense() const {
        return dense_;
      }

      template <typename Index>
      typename std::enable_if<btas::is_index<Index>::value, value_type>::type
      operator()(const Index& index) const {
        value_type o = 0;
        const auto end = this->rank();
        for(auto i = 0ul; i != end; ++i)
          o += *(index.begin() + i) * *(this->weight_.begin() + i);

        return o - offset_;
      }

    private:

      template <typename Index1,
                typename Index2,
                class = typename std::enable_if<btas::is_index<Index1>::value && btas::is_index<Index2>::value>::type
               >
      void init(const Index1& lobound,
                const Index2& upbound) {
        using btas::rank;
        auto n = rank(lobound);
        if (n == 0) return;

        value_type volume = 1;
        offset_ = 0;
        weight_ = array_adaptor<weight_type>::construct(n);

        // Compute range data
        if (order == CblasRowMajor) {
          for(int i = n - 1; i >= 0; --i) {
            weight_[i] = volume;
            auto li = *(lobound.begin() + i);
            auto ui = *(upbound.begin() + i);
            offset_ += li * volume;
            volume *= (ui - li);
          }
        }
        else {
          for(auto i = 0; i != n; ++i) {
            weight_[i] = volume;
            auto li = *(lobound.begin() + i);
            auto ui = *(upbound.begin() + i);
            offset_ += li * volume;
            volume *= (ui - li);
          }
        }
        dense_ = true;
      }

      template <typename Index1,
                typename Index2,
                typename Weight,
                class = typename std::enable_if<btas::is_index<Index1>::value &&
                                                btas::is_index<Index2>::value &&
                                                btas::is_index<Weight>::value>::type
               >
      void init(const Index1& lobound,
                const Index2& upbound,
                const Weight& weight) {
        using btas::rank;
        auto n = rank(lobound);
        if (n == 0) return;

        value_type volume = 1;
        offset_ = 0;
        weight_ = array_adaptor<weight_type>::construct(n);
        std::copy(weight.begin(), weight.end(), weight_.begin());

        // Compute offset and check whether dense
        dense_ = true;
        weight_type tmpweight = array_adaptor<weight_type>::construct(n);
        if (order == CblasRowMajor) {
          for(int i = n - 1; i >= 0; --i) {
            tmpweight[i] = volume;
            dense_ &= (tmpweight[i] == weight_[i]);
            auto li = *(lobound.begin() + i);
            auto ui = *(upbound.begin() + i);
            offset_ += li * weight_[i];
            volume *= (ui - li);
          }
        }
        else {
          for(auto i = 0; i != n; ++i) {
            tmpweight[i] = volume;
            dense_ &= (tmpweight[i] == weight_[i]);
            auto li = *(lobound.begin() + i);
            auto ui = *(upbound.begin() + i);
            offset_ += li * weight_[i];
            volume *= (ui - li);
          }
        }
      }

      weight_type weight_; // weight of each dimension (stride in the language of NumPy)
      value_type offset_; // lobound.weight => ordinal(index) = index.weight - offset
      bool dense_; // whether index iterator traverses a dense sequence of ordinals
  };

}

/// Range output operator

/// \param os The output stream that will be used to print \c r
/// \param r The range to be printed
/// \return A reference to the output stream
template <CBLAS_ORDER _Order,
          typename _Index>
inline std::ostream& operator<<(std::ostream& os, const btas::BoxOrdinal<_Order,_Index>& ord) {
  btas::array_adaptor<typename btas::BoxOrdinal<_Order,_Index>::weight_type>::print(ord.weight(), os);
  return os;
}


#endif /* BTAS_ORDINAL_H_ */
