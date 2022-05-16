#ifndef __BTAS_PERMUTE_H
#define __BTAS_PERMUTE_H 1

#include <algorithm>
#include <iterator>
#include <type_traits>

#include <btas/types.h>
#include <btas/util/resize.h>

#include <btas/tensor.h>
#include <btas/tensor_traits.h>
#include <btas/index_traits.h>

namespace btas {

  /// permute \c X using permutation \c p given in the preimage ("from") convention, write result to \c Y
  template<class _TensorX, typename _Permutation, class _TensorY,
           class = typename std::enable_if<is_boxtensor<_TensorX>::value &&
                                           is_index<_Permutation>::value &&
                                           is_boxtensor<_TensorY>::value
                                          >::type
          >
  void
  permute(const _TensorX& X, const _Permutation& p, _TensorY& Y) {
    const auto& r = X.range();
    using range_type = std::decay_t<decltype(r)>;
    constexpr const bool r_is_permutable = range_traits<range_type>::is_general_layout;

    auto do_perm = [](auto&& X, auto&& Y, auto&& pr) {
      Y.resize(pr);
      const auto itrX = std::begin(X);
      auto itrY = std::begin(Y);
      for (auto && i : Y.range()) {
        *itrY = *(itrX + pr.ordinal(i));
        ++itrY;
      }
    };
    if (r_is_permutable)
      do_perm(X, Y, permute(r, p));
    else {
      do_perm(X, Y, permute(btas::Range(r.lobound(), r.upbound(), r.stride()), p));
    }
  }

  /// permute \c X using permutation \c p given in the preimage ("from") convention, write result to \c Y
  template<class _TensorX, class _TensorY, typename _T,
           class = typename std::enable_if<is_boxtensor<_TensorX>::value &&
                                           is_boxtensor<_TensorY>::value
                                          >::type
          >
  void permute(const _TensorX& X, std::initializer_list<_T> pi, _TensorY& Y) {
      permute(X, btas::DEFAULT::index<_T>(pi) , Y);
  }

  /// permute \c X annotated with \c aX into \c Y annotated with \c aY
  /// \tparam _AnnotationX a container type
  /// \tparam _AnnotationY a container type
  template<class _TensorX, typename _AnnotationX, class _TensorY, typename _AnnotationY,
           class = typename std::enable_if<is_boxtensor<_TensorX>::value &&
                                           is_boxtensor<_TensorY>::value &&
                                           is_container<_AnnotationX>::value &&
                                           is_container<_AnnotationY>::value>::type>
  void permute(const _TensorX& X, const _AnnotationX& aX,
                     _TensorY& Y, const _AnnotationY& aY) {

   const auto Xrank = rank(X);

   // check rank
   assert(Xrank == rank(aX) && Xrank == rank(aY));

   // case: doesn't need to permute
   if (std::equal(std::begin(aX), std::end(aX), std::begin(aY)))
   {
      Y = X; return;
   }

   {
      // validate aX
      auto aX_sorted = aX;
      std::sort(std::begin(aX_sorted), std::end(aX_sorted));
      assert(
          std::unique(std::begin(aX_sorted), std::end(aX_sorted)) == std::end(aX_sorted));

      // validate aY
      auto aY_sorted = aY;
      std::sort(std::begin(aY_sorted), std::end(aY_sorted));
      assert(
          std::unique(std::begin(aY_sorted), std::end(aY_sorted)) == std::end(aY_sorted));

      // and aX against aY
      assert(std::equal(std::begin(aX_sorted), std::end(aX_sorted), std::begin(aY_sorted)));
    }

   // calculate permutation

   btas::DEFAULT::index<size_t> prm(Xrank);

   const auto first = std::begin(aX);
   const auto last  = std::end(aX);
   auto aY_iter = std::begin(aY);
   for(size_t i = 0; i < Xrank; ++i, ++aY_iter)
   {
      auto found = std::find(std::begin(aX), std::end(aX), *aY_iter);
      assert(found != last);
      prm[i] = std::distance(first, found);
   }

   // call permute
   permute(X, prm, Y);
}

  /// permute \c X annotated with \c aX into \c Y annotated with \c aY
  template<class _TensorX, class _TensorY, typename _T,
           class = typename std::enable_if<is_boxtensor<_TensorX>::value &&
                                           is_boxtensor<_TensorY>::value
                                          >::type
          >
  void permute(const _TensorX& X, std::initializer_list<_T> aX,
                     _TensorY& Y, std::initializer_list<_T> aY) {
      permute(X, btas::DEFAULT::index<_T>(aX), Y, btas::varray<_T>(aY));
  }

} // namespace btas

#endif // __BTAS_PERMUTE_H
