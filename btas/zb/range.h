/*
 * zb/range.h
 *
 *  Slimmed-down version of RangeNd for zero-based indexing apps. State is just (extent[], rank); lobound is
 *  structurally zero, upbound = extent, strides are derived row-major on demand.
 *
 *  sizeof(zb::RangeNd<6, int16_t, int32_t>) == 14 (vs ~304 for btas::Range).
 */

#ifndef BTAS_ZB_RANGE_H_
#define BTAS_ZB_RANGE_H_

#include <btas/fwd.h>

#include <btas/error.h>
#include <btas/index_traits.h>
#include <btas/range_iterator.h>
#include <btas/range_traits.h>
#include <btas/serialization.h>
#include <btas/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <type_traits>

namespace btas {
namespace zb {

/// Packed zero-based index/extent vector. Stores at most \c MaxRank entries of
/// type \c Int plus a 1-byte size; size <= MaxRank invariant is enforced.
///
/// Models the TWG.Index concept (\c btas::is_index) — exposes nested
/// \c value_type plus member \c begin / \c end / \c operator[] / \c size .
template <std::size_t MaxRank, typename Int>
class index {
 public:
  using value_type = Int;
  using size_type = std::size_t;
  using reference = Int&;
  using const_reference = const Int&;
  using pointer = Int*;
  using const_pointer = const Int*;
  using iterator = Int*;
  using const_iterator = const Int*;

  static constexpr size_type max_size_v = MaxRank;

  constexpr index() noexcept : data_{}, size_(0) {}

  /// fixed-size construction with default-initialized elements
  constexpr explicit index(size_type n) : data_{}, size_(check_size(n)) {}

  /// fixed-size construction filled with \p v
  constexpr index(size_type n, Int v) : data_{}, size_(check_size(n)) {
    for (size_type i = 0; i < size_; ++i) data_[i] = v;
  }

  template <typename T>
  constexpr index(std::initializer_list<T> il) : data_{}, size_(check_size(il.size())) {
    size_type i = 0;
    for (auto v : il) data_[i++] = static_cast<Int>(v);
  }

  /// from any iterable container (size must be <= MaxRank)
  template <typename C,
            typename = std::enable_if_t<is_container<std::decay_t<C>>::value &&
                                        !std::is_same_v<std::decay_t<C>, index>>>
  constexpr index(const C& c) : data_{}, size_(0) {
    using std::begin;
    using std::end;
    auto first = begin(c);
    auto last = end(c);
    size_type n = 0;
    for (auto it = first; it != last; ++it) ++n;
    size_ = check_size(n);
    size_type i = 0;
    for (auto it = first; it != last; ++it) data_[i++] = static_cast<Int>(*it);
  }

  constexpr size_type size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }
  static constexpr size_type max_size() noexcept { return MaxRank; }

  constexpr reference operator[](size_type i) noexcept { return data_[i]; }
  constexpr const_reference operator[](size_type i) const noexcept { return data_[i]; }

  constexpr reference at(size_type i) { BTAS_ASSERT(i < size_); return data_[i]; }
  constexpr const_reference at(size_type i) const { BTAS_ASSERT(i < size_); return data_[i]; }

  constexpr iterator begin() noexcept { return data_.data(); }
  constexpr iterator end() noexcept { return data_.data() + size_; }
  constexpr const_iterator begin() const noexcept { return data_.data(); }
  constexpr const_iterator end() const noexcept { return data_.data() + size_; }
  constexpr const_iterator cbegin() const noexcept { return data_.data(); }
  constexpr const_iterator cend() const noexcept { return data_.data() + size_; }

  constexpr pointer data() noexcept { return data_.data(); }
  constexpr const_pointer data() const noexcept { return data_.data(); }

  /// resize to \p n (elements past the old size are value-initialized)
  constexpr void resize(size_type n) {
    auto new_size = check_size(n);
    for (size_type i = size_; i < new_size; ++i) data_[i] = Int{};
    size_ = new_size;
  }

  friend constexpr bool operator==(const index& a, const index& b) noexcept {
    if (a.size_ != b.size_) return false;
    for (std::uint8_t i = 0; i < a.size_; ++i)
      if (a.data_[i] != b.data_[i]) return false;
    return true;
  }
  friend constexpr bool operator!=(const index& a, const index& b) noexcept {
    return !(a == b);
  }

 private:
  static constexpr std::uint8_t check_size(size_type n) {
    BTAS_ASSERT(n <= MaxRank);
    return static_cast<std::uint8_t>(n);
  }

  std::array<Int, MaxRank> data_;
  std::uint8_t size_;
};

/// Lightweight value type returned by \c RangeNd::ordinal() . Synthesizes
/// strides from extent at construction; offset is always 0 and the range is
/// always contiguous, by construction.
template <std::size_t MaxRank, typename Ord>
class ordinal_view {
 public:
  using value_type = Ord;
  using stride_type = std::array<Ord, MaxRank>;

  ordinal_view() noexcept : stride_{}, rank_(0) {}

  template <typename Extents>
  ordinal_view(const Extents& ext, std::size_t rank) : stride_{}, rank_(rank) {
    using std::cbegin;
    auto it = cbegin(ext);
    Ord vol{1};
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(rank) - 1; i >= 0; --i) {
      stride_[i] = vol;
      vol *= static_cast<Ord>(*(it + i));
    }
  }

  std::size_t rank() const noexcept { return rank_; }
  const stride_type& stride() const noexcept { return stride_; }
  const Ord* stride_data() const noexcept { return stride_.data(); }
  constexpr Ord offset() const noexcept { return Ord{0}; }
  constexpr bool contiguous() const noexcept { return true; }

  template <typename Index>
  std::enable_if_t<is_index<Index>::value, Ord>
  operator()(const Index& idx) const {
    BTAS_ASSERT(static_cast<std::size_t>(idx.size()) == rank_);
    using std::cbegin;
    Ord o{0};
    auto it = cbegin(idx);
    for (std::size_t i = 0; i < rank_; ++i)
      o += static_cast<Ord>(*(it + i)) * stride_[i];
    return o;
  }

 private:
  stride_type stride_;
  std::size_t rank_;
};

/// Zero-based row-major N-dim range optimized for applications with zero-based indexing.
///
/// \tparam MaxRank static cap on rank (default 6)
/// \tparam Ext     per-dim extent integer type (default int16_t)
/// \tparam Ord     ordinal integer type (default int32_t)
template <std::size_t MaxRank = 6,
          typename Ext = std::int16_t,
          typename Ord = std::int32_t>
class RangeNd {
 public:
  static_assert(MaxRank > 0 && MaxRank < 256, "MaxRank must lie in (0, 256)");
  static_assert(std::is_integral_v<Ext>, "Ext must be an integer type");
  static_assert(std::is_integral_v<Ord> && std::is_signed_v<Ord>,
                "Ord must be a signed integer type");

  static constexpr ::blas::Layout order = ::blas::Layout::RowMajor;
  static constexpr std::size_t max_rank = MaxRank;

  using extent_type = index<MaxRank, Ext>;
  using index_type = extent_type;
  using index1_type = Ext;
  using index_element_type = Ext;
  using extent_element_type = Ext;
  using ordinal_type = Ord;
  using size_type = std::size_t;

  using value_type = index_type;
  using reference = index_type&;
  using const_reference = const index_type&;

  using iterator = btas::RangeIterator<index_type, RangeNd>;
  using const_iterator = iterator;
  friend class btas::RangeIterator<index_type, RangeNd>;

  /// Default constructor: rank-0 range.
  constexpr RangeNd() noexcept = default;

  /// Construct from an extent container (rank inferred from size).
  template <typename C,
            typename = std::enable_if_t<is_index<std::decay_t<C>>::value &&
                                        !std::is_same_v<std::decay_t<C>, RangeNd>>>
  RangeNd(const C& ext) : extent_(ext) {}

  /// Construct from an initializer list of extents.
  template <typename T,
            typename = std::enable_if_t<std::is_integral_v<T>>>
  RangeNd(std::initializer_list<T> il) : extent_(il) {}

  /// Construct from a pack of integer extents (>=2 to avoid clashing with
  /// the container-taking constructor; pass a one-element \c {e0} for rank-1).
  template <typename I0, typename I1, typename... Is,
            typename = std::enable_if_t<
                std::is_integral_v<I0> && std::is_integral_v<I1> &&
                (std::is_integral_v<Is> && ...)>>
  RangeNd(I0 e0, I1 e1, Is... es)
      : extent_({static_cast<Ext>(e0), static_cast<Ext>(e1),
                 static_cast<Ext>(es)...}) {}

  /// Construct from lobound/upbound pair. \c lobound must be all zeros
  /// (zero-based ranges); \c upbound becomes the extent. Useful because
  /// downstream code such as @c btas::Tensor 's @c (range, storage) ctor
  /// instantiates @c range_type(lobound, upbound) even when the runtime
  /// branch would not take that path.
  template <typename Lo, typename Up,
            typename = std::enable_if_t<is_index<std::decay_t<Lo>>::value &&
                                        is_index<std::decay_t<Up>>::value>>
  RangeNd(const Lo& lobound, const Up& upbound) : extent_(upbound) {
    (void)lobound;
    using std::cbegin;
    using std::cend;
    BTAS_ASSERT(std::all_of(cbegin(lobound), cend(lobound),
                       [](auto v) { return v == 0; }) &&
           "btas::zb::RangeNd: lobound must be all zeros");
  }

  //
  // Rank, extent, lo/up bounds
  //

  std::size_t rank() const noexcept { return extent_.size(); }

  /// Volume; matches \c btas::BaseRangeNd::area() which returns 0 for rank 0.
  size_type area() const noexcept {
    if (extent_.size() == 0) return 0;
    size_type v = 1;
    for (std::size_t i = 0; i < extent_.size(); ++i)
      v *= static_cast<size_type>(extent_[i]);
    return v;
  }
  size_type volume() const noexcept { return area(); }

  const extent_type& extent() const noexcept { return extent_; }
  Ext extent(std::size_t i) const noexcept { return extent_[i]; }
  const Ext* extent_data() const noexcept { return extent_.data(); }

  /// Lower bound: always zeros. Returns by value (small, fixed-size object).
  index_type lobound() const { return index_type(rank(), Ext{0}); }
  Ext lobound(std::size_t) const noexcept { return Ext{0}; }
  const Ext* lobound_data() const noexcept { return zero_buffer(); }

  /// Upper bound: equal to extent (zero-based range).
  const extent_type& upbound() const noexcept { return extent_; }
  Ext upbound(std::size_t i) const noexcept { return extent_[i]; }
  const Ext* upbound_data() const noexcept { return extent_.data(); }

  //
  // Ordinal mapping (synthesized row-major; nothing stored)
  //

  ordinal_view<MaxRank, Ord> ordinal() const {
    return ordinal_view<MaxRank, Ord>(extent_, rank());
  }

  template <typename I>
  std::enable_if_t<is_index<I>::value, Ord> ordinal(const I& idx) const {
    BTAS_ASSERT(static_cast<std::size_t>(idx.size()) == rank());
    using std::cbegin;
    auto it = cbegin(idx);
    const auto r = rank();
    Ord o{0};
    Ord vol{1};
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(r) - 1; i >= 0; --i) {
      o += static_cast<Ord>(*(it + i)) * vol;
      vol *= static_cast<Ord>(extent_[i]);
    }
    return o;
  }

  //
  // Iteration
  //

  const_iterator begin() const { return const_iterator(lobound(), this); }
  const_iterator end() const { return const_iterator(extent_, this); }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }

  /// Row-major in-place increment used by \c RangeIterator . After the final
  /// valid index, idx == upbound() (== extent_), which matches \c end() .
  void increment(index_type& idx) const {
    const auto r = rank();
    if (r == 0) return;
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(r) - 1; d >= 0; --d) {
      ++idx[d];
      if (idx[d] < extent_[d]) return;
      idx[d] = Ext{0};
    }
    for (std::size_t d = 0; d < r; ++d) idx[d] = extent_[d];
  }

  //
  // Equality, swap
  //

  friend bool operator==(const RangeNd& a, const RangeNd& b) noexcept {
    return a.extent_ == b.extent_;
  }
  friend bool operator!=(const RangeNd& a, const RangeNd& b) noexcept {
    return !(a == b);
  }

  void swap(RangeNd& other) noexcept {
    using std::swap;
    swap(extent_, other.extent_);
  }

 private:
  /// Static MaxRank-sized zero buffer; \c lobound_data() returns a pointer
  /// into it. Callers iterate only the first \c rank() bytes.
  static const Ext* zero_buffer() noexcept {
    static const std::array<Ext, MaxRank> z{};
    return z.data();
  }

  extent_type extent_{};
};

template <std::size_t MaxRank, typename Ext, typename Ord>
inline void swap(RangeNd<MaxRank, Ext, Ord>& a,
                 RangeNd<MaxRank, Ext, Ord>& b) noexcept {
  a.swap(b);
}

}  // namespace zb

//
// Trait specializations placing zb::RangeNd into the BTAS Range concept.
//

template <std::size_t MaxRank, typename Ext, typename Ord>
struct range_traits<zb::RangeNd<MaxRank, Ext, Ord>> {
  static constexpr ::blas::Layout order = ::blas::Layout::RowMajor;
  using index_type = typename zb::RangeNd<MaxRank, Ext, Ord>::index_type;
  using ordinal_type = Ord;
  static constexpr bool is_general_layout = false;
};

template <std::size_t MaxRank, typename Ext, typename Ord>
class boxrange_iteration_order<zb::RangeNd<MaxRank, Ext, Ord>> {
 public:
  enum {
    row_major = boxrange_iteration_order<void>::row_major,
    other = boxrange_iteration_order<void>::other,
    column_major = boxrange_iteration_order<void>::column_major
  };
  static constexpr int value = row_major;
};

}  // namespace btas

//
// MADNESS archive load/store specializations. Includes only the rank + extents;
// the ordinal is fully derivable from extent. Caller must have included
// <madness/world/archive.h> before instantiating these.
//
namespace madness {
namespace archive {

template <class Archive, std::size_t MaxRank, typename Ext, typename Ord>
struct ArchiveLoadImpl<Archive, btas::zb::RangeNd<MaxRank, Ext, Ord>> {
  static inline void load(const Archive& ar,
                          btas::zb::RangeNd<MaxRank, Ext, Ord>& r) {
    std::uint8_t rank{};
    ar& rank;
    typename btas::zb::RangeNd<MaxRank, Ext, Ord>::extent_type ext(
        static_cast<std::size_t>(rank));
    for (std::uint8_t i = 0; i < rank; ++i) ar& ext[i];
    r = btas::zb::RangeNd<MaxRank, Ext, Ord>(ext);
  }
};

template <class Archive, std::size_t MaxRank, typename Ext, typename Ord>
struct ArchiveStoreImpl<Archive, btas::zb::RangeNd<MaxRank, Ext, Ord>> {
  static inline void store(const Archive& ar,
                           const btas::zb::RangeNd<MaxRank, Ext, Ord>& r) {
    const std::uint8_t rank = static_cast<std::uint8_t>(r.rank());
    ar& rank;
    for (std::uint8_t i = 0; i < rank; ++i) ar& r.extent(i);
  }
};

}  // namespace archive
}  // namespace madness

#endif  // BTAS_ZB_RANGE_H_
