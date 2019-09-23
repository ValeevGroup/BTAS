//
// Created by efv on 9/23/19.
//

#ifndef BTAS_BTAS_FWD_H
#define BTAS_BTAS_FWD_H

#include <iosfwd>

#include <btas/types.h>
#include <btas/defaults.h>

namespace btas {

    /// test _Index conforms the TWG.Index concept
    /// check only value_type and operator[]
    template<typename ...>
    class is_index;

    /// BoxOrdinal is an implementation detail of BoxRange.
    /// It maps the index to its ordinal value. It also knows whether
    /// the map is contiguous (i.e. whether adjacent indices have adjacent ordinal
    /// values).
    template <CBLAS_ORDER _Order = CblasRowMajor,
            typename _Index = btas::DEFAULT::index_type
    >
    class BoxOrdinal;

    template <CBLAS_ORDER _Order,
            typename _Index>
    std::ostream& operator<<(std::ostream&, const BoxOrdinal<_Order,_Index>&);

        /// RangeNd extends BaseRangeNd to compute ordinals, as specified by \c _Ordinal .
    /// It conforms to the \ref sec_TWG_Range_Concept_Range_Box "TWG.BoxRange" concept.
    template <CBLAS_ORDER _Order = CblasRowMajor,
            typename _Index = btas::DEFAULT::index_type,
            typename _Ordinal = btas::BoxOrdinal<_Order,_Index>
    >
    class RangeNd;

    template <CBLAS_ORDER _Order,
            typename _Index,
            typename _Ordinal>
    std::ostream& operator<<(std::ostream&, const RangeNd<_Order,_Index, _Ordinal>&);


        namespace DEFAULT {
        using range = btas::RangeNd<>;
    }  // namespace DEFAULT

    /// checks _Tensor meets the TWG.Tensor concept requirements
    /// checks only value_type, range_type, storage_type, and rank() member
    template<class _Tensor>
    class is_tensor;

    /// checks _Tensor meets the TWG.BoxTensor concept requirements
    template<class _Tensor>
    class is_boxtensor;

    template<typename _T,
            class _Range = btas::DEFAULT::range,
            class _Storage = btas::DEFAULT::storage<_T>
    >
    class Tensor;

    template <typename _T,
            class _Range = btas::DEFAULT::range,
            class _Storage = btas::DEFAULT::storage<_T>>
    std::ostream& operator<<(std::ostream&, const Tensor<_T,_Range, _Storage>&);

}

#endif //BTAS_BTAS_FWD_H
