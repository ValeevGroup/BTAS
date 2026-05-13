#include "test.h"

#include <array>
#include <cstdint>
#include <vector>

#include "btas/zb/range.h"

using btas::zb::RangeNd;
// NB: do NOT `using btas::zb::index` at file scope — macOS POSIX <strings.h>
// declares ::index(const char*, int) and the names collide.

// Sizeof contract: this is the whole point of zb::RangeNd. Default
// MaxRank=6 / Ext=int16_t / Ord=int32_t must fit in 14 bytes (12 B extents +
// 1 B size, alignment 2).
static_assert(sizeof(RangeNd<>) == 14, "zb::RangeNd<> must be 14 bytes");
static_assert(alignof(RangeNd<>) == 2, "zb::RangeNd<> must align to 2");

// Index concept membership: TWG.Index is what btas::Tensor and the
// expression layer check via SFINAE.
static_assert(btas::is_index<typename RangeNd<>::index_type>::value,
              "zb::index must model btas::is_index");
static_assert(btas::is_boxrange<RangeNd<>>::value,
              "zb::RangeNd must model btas::is_boxrange");
static_assert(btas::boxrange_iteration_order<RangeNd<>>::value ==
                  btas::boxrange_iteration_order<void>::row_major,
              "zb::RangeNd must be row-major");

TEST_CASE("zb::index basics") {
  using Idx = btas::zb::index<6, std::int16_t>;

  SECTION("default") {
    Idx a;
    CHECK(a.size() == 0);
    CHECK(a.empty());
    CHECK(a == Idx{});
  }

  SECTION("initializer list") {
    Idx a{2, 3, 4};
    CHECK(a.size() == 3);
    CHECK(a[0] == 2);
    CHECK(a[1] == 3);
    CHECK(a[2] == 4);
  }

  SECTION("from container") {
    std::vector<int> v{5, 6};
    Idx a(v);
    CHECK(a.size() == 2);
    CHECK(a[0] == 5);
    CHECK(a[1] == 6);
  }

  SECTION("equality") {
    CHECK(Idx{1, 2} == Idx{1, 2});
    CHECK(Idx{1, 2} != Idx{1, 2, 3});
    CHECK(Idx{1, 2} != Idx{2, 1});
  }
}

TEST_CASE("zb::RangeNd construction and accessors") {
  SECTION("default is rank-0, area()==0") {
    RangeNd<> r;
    CHECK(r.rank() == 0);
    CHECK(r.area() == 0);  // matches btas::BaseRangeNd::area() convention
  }

  SECTION("from variadic extents") {
    RangeNd<> r(2, 3, 4);
    CHECK(r.rank() == 3);
    CHECK(r.area() == 24);
    CHECK(r.extent(0) == 2);
    CHECK(r.extent(1) == 3);
    CHECK(r.extent(2) == 4);
  }

  SECTION("from initializer list") {
    RangeNd<> r{5, 7};
    CHECK(r.rank() == 2);
    CHECK(r.area() == 35);
  }

  SECTION("lobound is zeros, upbound is extent") {
    RangeNd<> r(2, 3, 4);
    auto lo = r.lobound();
    CHECK(lo.size() == 3);
    CHECK(lo[0] == 0);
    CHECK(lo[1] == 0);
    CHECK(lo[2] == 0);
    CHECK(r.upbound() == r.extent());
    CHECK(r.upbound_data() == r.extent_data());
  }

  SECTION("lobound_data points to MaxRank zeros") {
    RangeNd<> r(2, 3);
    auto* p = r.lobound_data();
    for (std::size_t i = 0; i < RangeNd<>::max_rank; ++i) CHECK(p[i] == 0);
  }
}

TEST_CASE("zb::RangeNd ordinal mapping is row-major contiguous") {
  RangeNd<> r(2, 3, 4);

  SECTION("formula") {
    // row-major: ord(i,j,k) = i*(3*4) + j*4 + k
    using idx_t = RangeNd<>::index_type;
    CHECK(r.ordinal(idx_t{0, 0, 0}) == 0);
    CHECK(r.ordinal(idx_t{0, 0, 1}) == 1);
    CHECK(r.ordinal(idx_t{0, 1, 0}) == 4);
    CHECK(r.ordinal(idx_t{1, 0, 0}) == 12);
    CHECK(r.ordinal(idx_t{1, 2, 3}) == 23);
  }

  SECTION("ordinal_view exposes strides and is contiguous") {
    auto ov = r.ordinal();
    CHECK(ov.rank() == 3);
    CHECK(ov.contiguous());
    CHECK(ov.offset() == 0);
    CHECK(ov.stride()[0] == 12);
    CHECK(ov.stride()[1] == 4);
    CHECK(ov.stride()[2] == 1);
    using idx_t = RangeNd<>::index_type;
    CHECK(ov(idx_t{1, 2, 3}) == 23);
  }
}

TEST_CASE("zb::RangeNd iteration covers volume in row-major order") {
  RangeNd<> r(2, 3);
  std::size_t count = 0;
  std::int32_t expected_ordinal = 0;
  for (auto it = r.begin(); it != r.end(); ++it, ++count, ++expected_ordinal) {
    CHECK(r.ordinal(*it) == expected_ordinal);
  }
  CHECK(count == r.area());
}

TEST_CASE("zb::RangeNd equality and swap") {
  RangeNd<> a(2, 3, 4);
  RangeNd<> b(2, 3, 4);
  RangeNd<> c(2, 3, 5);

  CHECK(a == b);
  CHECK(a != c);

  using std::swap;
  swap(a, c);
  CHECK(a == RangeNd<>(2, 3, 5));
  CHECK(c == RangeNd<>(2, 3, 4));
}

TEST_CASE("zb::RangeNd with non-default template parameters") {
  using R4 = RangeNd<::blas::Layout::RowMajor, std::int32_t, std::int64_t, 4>;
  static_assert(sizeof(R4) == 20, "expected packed size for MaxRank=4, int32");
  static_assert(R4::max_rank == 4);
  R4 r(10, 20, 30);
  CHECK(r.rank() == 3);
  CHECK(r.area() == 6000);
  CHECK(r.ordinal(typename R4::index_type{1, 2, 3}) ==
        1 * 20 * 30 + 2 * 30 + 3);
}

TEST_CASE("zb::RangeNd column-major layout") {
  using RC = RangeNd<::blas::Layout::ColMajor>;
  RC r(10, 20, 30);
  CHECK(r.rank() == 3);
  CHECK(r.area() == 6000);
  // col-major: ordinal = i0 + i1*ext0 + i2*ext0*ext1
  CHECK(r.ordinal(typename RC::index_type{1, 2, 3}) ==
        1 + 2 * 10 + 3 * 10 * 20);
  // strides[0]=1, strides[1]=ext0=10, strides[2]=ext0*ext1=200
  auto s = r.stride();
  CHECK(s[0] == 1);
  CHECK(s[1] == 10);
  CHECK(s[2] == 200);
  // Iteration covers volume in column-major order.
  std::size_t count = 0;
  typename RC::index_type prev;
  for (auto&& idx : r) {
    (void)idx;
    ++count;
  }
  CHECK(count == r.area());
}
