#include "test.h"

#include <iostream>
#include <random>

#include "btas/tensorview.h"
#include "btas/tensor.h"
#include "btas/tensor_func.h"
// serialization of TensorView is disabled
#if 0
#include <fstream>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#endif

using std::cout;
using std::endl;

using btas::Range;
using btas::Tensor;
using DTensor = Tensor<double>;
using btas::TensorView;
using namespace btas;

template <typename T> struct TD;
template <typename T>
void BindToConstRefAccess(T&& x) {
}

static std::ostream&
operator<<(std::ostream& s, const DTensor& X) {
  for (auto i : X.range())
    s << i << " " << X(i) << "\n";
  return s;
}

// Set the elements of a Tensor T such that
// T(i,j,k) = 1ijk
// assuming individual dimensions are all less than 10.
// (The 1 in front is just a placeholder.)
void static fillEls(DTensor& T) {
  if (T.rank() == 0)
    return;
  const double base = pow(10., T.rank());
  const size_t max_ii = T.rank() - 1;
  for (auto I : T.range()) {
    double &val = T(I);
    val = base;
    for (size_t ii = 0; ii <= max_ii; ++ii) {
      val += I[ii] * pow(10., max_ii - ii);
    }
  }
}

TEST_CASE("TensorView constructors") {

  DTensor T0(2, 3, 4);
  fillEls(T0);

  auto& T0_ref = T0;
  const auto& T0_cref = T0;

  SECTION("default ctor for TensorView<double>") {
    TensorView<double> T0v;
  }

  SECTION("TensorView<double> directly from Tensor<double>") {
    TensorView<double> T0v(T0);
    CHECK(T0v == T0);
  }

  SECTION("TensorView<double> directly from Tensor<double>&") {
    TensorView<double> T0v(T0_ref);
    CHECK(T0v == T0);
  }

  SECTION("TensorView<double> cannot be constructed directly from const Tensor<double>&") {
    //TensorView<double> T0v(T0_cref); // compile error: nonconst view from const Tensor
  }

  SECTION("TensorConstView<double> directly from Tensor<double>") {
    TensorConstView<double> T0v(T0);
    CHECK(T0v == T0);
  }

  SECTION("TensorConstView<double> directly from Tensor<double>&") {
    TensorConstView<double> T0v(T0_ref);
    CHECK(T0v == T0);
  }

  SECTION("TensorConstView<double> directly from const Tensor<double>&") {
    TensorConstView<double> T0v(T0_cref);
    CHECK(T0v == T0);
  }

  SECTION("TensorView<double> using make_view from Tensor<double>") {
    auto T0vd = make_view(T0);
    CHECK(T0vd == T0);
  }

  SECTION("TensorView<double> using make_cview from Tensor<double>") {
    auto T0cvd = make_cview(T0);
    CHECK(T0cvd == T0);
  }

  SECTION("TensorView<float> using make_view from Tensor<double>") {
    auto T0vf = make_view<float>(T0);
    CHECK(T0vf == T0);
  }

  SECTION("TensorView<float> using make_cview from Tensor<double>") {
    auto T0vf = make_cview<float>(T0);
    CHECK(T0vf == T0);
  }

  SECTION("TensorRWView<double> using make_rwview from Tensor<double>") {
    auto T0vd = make_rwview(T0);
    CHECK(T0vd == T0);
  }

  SECTION("TensorRWView<double> using make_rwview from Tensor<double>") {
    auto T0cvd = make_rwview(T0, false);
    CHECK(T0cvd == T0);
  }

  SECTION("TensorRWView<double> using make_rwview from TensorRWView(read-only) throws") {
    auto T0v = make_rwview(T0, false);
    // TODO need to specialize make_*view to TensorView
    CHECK_THROWS_AS( make_rwview(T0v), btas::exception);
  }

  SECTION("TensorRWView<float> using make_rwview from Tensor<double>") {
    auto T0vf = make_rwview<float>(T0);
    CHECK(T0vf == T0);
  }

  SECTION("TensorRWView<float> using make_rwview from Tensor<double>") {
    auto T0vf = make_rwview<float>(T0, false);
    CHECK(T0vf == T0);
  }


  SECTION("TensorView<double> using make_view from permuted Range + Storage") {
    const auto& T0_cref = T0;
    auto prange0 = permute(T0.range(), {2,1,0});

    // read only views
    const auto T0cvr = make_view(prange0, T0_cref.storage());
    bool tensorview_is_readonly = true;
    for(size_t i0=0; i0< T0.extent(0);i0++)
      for(size_t i1=0; i1< T0.extent(1);i1++)
        for(size_t i2=0; i2< T0.extent(2);i2++)
          tensorview_is_readonly = tensorview_is_readonly && (T0cvr(i2,i1,i0) == T0(i0,i1,i2));
    CHECK(tensorview_is_readonly);

    auto T0cv = make_cview(prange0, T0.storage());
    CHECK(T0cv == T0cvr);
    auto T0ncvr = make_view(prange0, T0_cref.storage());
    CHECK(T0ncvr == T0cvr);

    // readwrite view
    auto T0vw = make_view(prange0, T0.storage());
    CHECK(T0vw == T0cvr);
  }

} // TEST_CASE("TensorView constructors")

TEST_CASE("TensorView assignment") {

  DTensor T0(2, 3, 4);
  fillEls(T0);

  auto& T0_ref = T0;
  const auto& T0_cref = T0;
  TensorView<double> T0v(T0_ref);
  TensorConstView<double> T0cv(T0_cref);

  DTensor T2(3, 4);
  fillEls(T2);

  SECTION("TensorView<double> = Tensor<double>") {
    TensorView<double> T0v = T0;
    CHECK(T0v == T0);
  }

  SECTION("TensorView<double> = const Tensor<double>: compile-time error") {
    //TensorView<double> T0v = T0_cref;
  }

  SECTION("TensorView<double> = TensorView<double>") {
    TensorView<double> T0v2 = T0v;
    CHECK(T0v2 == T0);
  }

  SECTION("TensorView<double> = TensorConstView<double>: compile-time error") {
    //TensorView<double> T0v = T0cv; // compile-time error
  }

  SECTION("TensorConstView<double> = Tensor<double>") {
    TensorConstView<double> T0cv = T0;
    CHECK(T0cv == T0);
  }

  SECTION("TensorConstView<double> = const Tensor<double>") {
    TensorConstView<double> T0cv = T0_cref;
    CHECK(T0cv == T0);
  }

  SECTION("TensorConstView<double> = TensorView<double>") {
    TensorConstView<double> T0cv = T0v;
    CHECK(T0cv == T0);
  }

  SECTION("TensorConstView<double> = TensorConstView<double>: runtime error") {
    // TODO specialize TensorView (_Tensor& x) constructor for TensorView& x
    //TensorConstView<double> T0cv2 = T0cv;
  }

  SECTION("TensorRWView<double> = Tensor<double>") {
    TensorRWView<double> T0cv = T0;
    CHECK(T0cv == T0);
    CHECK(T0cv.writable() == true);
  }

  SECTION("TensorRWView<double> = const Tensor<double>") {
    TensorRWView<double> T0cv = T0_cref;
    CHECK(T0cv == T0);
    CHECK(T0cv.writable() == false);
  }

  SECTION("TensorRWView<double> = TensorView<double>") {
    TensorRWView<double> T0cv = T0v;
    CHECK(T0cv == T0);
    CHECK(T0cv.writable() == true);
  }

  SECTION("TensorRWView<double> = const TensorView<double>") {
    const auto& T0v_cref = T0v;
    TensorRWView<double> T0cv = T0v_cref;
    CHECK(T0cv == T0);
    CHECK(T0cv.writable() == false);
  }

  SECTION("TensorRWView<double> = TensorConstView<double>") {
    TensorRWView<double> T0cv2 = T0cv;
    CHECK(T0cv2 == T0);
    CHECK(T0cv.writable() == false);
  }

  SECTION("to Tensor using permute() -> TensorView") {
    //This is a regression test for bug #64
    //The following code was failing due
    //to a faulty implementation of Tensor operator=
    DTensor pT2;
    pT2 = permute(T2, {1,0});
    CHECK(pT2(1,0) == T2(0,1));
  }

} // TEST_CASE("TensorView assignment")

TEST_CASE("TensorView constness tracking") {

  DTensor T0(2, 3, 4);
  fillEls(T0);
  const auto& T0_cref = T0;
  auto& T0_ref = T0;

  SECTION("directly constructed TensorView is writable") {
    TensorView<double> T0v(T0);
    T0v(0,0,0) = 1.0;
    CHECK(T0v(0,0,0) == 1.0);
    CHECK(T0(0,0,0) == 1.0);
  }

  SECTION("make_view makes writable TensorView") {
    auto T0vd = make_view(T0);
    T0vd(0,0,0) = 1.0;
    CHECK(T0vd(0,0,0) == 1.0);
    CHECK(T0(0,0,0) == 1.0);
  }

  SECTION("make_cview makes read-only TensorView") {
    auto T0cvd = make_cview(T0);
    // T0cvd(0,0,0) == 1.0; // compile error : assignment to read-only value
    auto tensorview_elemaccess_returns_constdoubleref = std::is_same<decltype(T0cvd(0,0,0)),const double&>::value;
    CHECK(tensorview_elemaccess_returns_constdoubleref);// ensure operator() returns const ref
  }

  SECTION("make_rwview(Tensor) makes writable TensorView") {
    auto T0vd = make_rwview(T0);
    T0vd(0,0,0) = 1.0;
    CHECK(T0vd(0,0,0) == 1.0);
    CHECK(T0(0,0,0) == 1.0);
  }

  SECTION("make_rwview(Tensor,false) makes read-only TensorView") {
    auto T0vd = make_rwview(T0, false);
    CHECK_THROWS_AS( T0vd(0,0,0) = 1.0 , btas::exception);
  }

  SECTION("make_rwview(const Tensor) makes read-only TensorView") {
    auto T0v = make_rwview(T0_cref);
    CHECK_THROWS_AS( T0v(0,0,0) = 1.0 , btas::exception);
  }

  SECTION("make_rwview(const Tensor, false) makes read-only TensorView") {
    auto T0v = make_rwview(T0_cref, false);
    CHECK_THROWS_AS( T0v(0,0,0) = 1.0 , btas::exception);
  }

  SECTION("make_rwview(TensorView) makes writable TensorView") {
    auto T0v = make_view(T0);
    auto T0vv = make_rwview(T0v);
    T0vv(0,0,0) = 1.0;
    CHECK(T0vv(0,0,0) == 1.0);
    CHECK(T0(0,0,0) == 1.0);
  }

  SECTION("make_rwview(TensorView,false) makes read-only TensorView") {
    auto T0v = make_view(T0);
    auto T0vv = make_rwview(T0v, false);
    CHECK_THROWS_AS( T0vv(0,0,0) = 1.0 , btas::exception);
  }

  SECTION("make_rwview(const TensorView) makes read-only TensorView") {
    const auto T0v = make_view(T0);
    auto T0vv = make_rwview(T0v);
    CHECK_THROWS_AS( T0vv(0,0,0) = 1.0 , btas::exception);
  }

  SECTION("make_rwview(const TensorView, false) makes read-only TensorView") {
    const auto T0v = make_view(T0);
    auto T0vv = make_rwview(T0v, false);
    CHECK_THROWS_AS( T0vv(0,0,0) = 1.0 , btas::exception);
  }

  SECTION("make_rwview(TensorRWView(writable)) makes writable TensorView") {
    auto T0v = make_rwview(T0);
    auto T0vv = make_rwview(T0v);
    T0vv(0,0,0) = 1.0;
    CHECK(T0vv(0,0,0) == 1.0);
    CHECK(T0(0,0,0) == 1.0);
  }

  SECTION("make_rwview(const TensorRWView(writable)) makes read-only TensorView") {
    const auto T0v = make_rwview(T0);
    auto T0vv = make_rwview(T0v);
    CHECK_THROWS_AS( T0vv(0,0,0) = 1.0 , btas::exception);
  }

  SECTION("make_rwview(TensorConstView) makes read-only TensorView") {
    auto T0v = make_cview(T0);
    auto T0vv = make_rwview(T0v);
    CHECK_THROWS_AS( T0vv(0,0,0) = 1.0 , btas::exception);
  }

  SECTION("make_view<float> makes read-only TensorView") {
    auto T0vf = make_view<float>(T0);
    auto tensorview_elemaccess_returns_float = std::is_same<decltype(T0vf(0,0,0)),float>::value;
    CHECK(tensorview_elemaccess_returns_float); // ensure operator() returns float
  }

  SECTION("make_cview<float> makes read-only TensorView") {
    auto T0vf = make_cview<float>(T0);
    auto tensorview_elemaccess_returns_float = std::is_same<decltype(T0vf(0,0,0)),float>::value;
    CHECK(tensorview_elemaccess_returns_float); // ensure operator() returns float
  }

  SECTION("make_rwview<float> makes read-only TensorView") {
    auto T0vf = make_rwview<float>(T0);
    auto tensorview_elemaccess_returns_float = std::is_same<decltype(T0vf(0,0,0)),float>::value;
    CHECK(tensorview_elemaccess_returns_float); // ensure operator() returns float
  }

} // TEST_CASE("TensorView constness tracking")

TEST_CASE("TensorMap constructors") {

  double* ptr0 = new double[24];
  for(auto i=0; i!=24; ++i) ptr0[i] = i;
  const double* cptr0 = const_cast<const double*>(ptr0);

  SECTION("make_map<double> makes writable TensorMap from double*") {
    auto map = make_map(ptr0, Range{2, 3, 4});
    map(0,0,0) = 1.0;
    CHECK(map(0,0,0) == 1.0);
  }

  SECTION("make_map<double> makes read-only TensorMap from const double*") {
    auto map = make_map(cptr0, Range{2, 3, 4});
    auto map_elemaccess_returns_constref = std::is_same<decltype(map(0,0,0)),const double&>::value;
    CHECK(map_elemaccess_returns_constref); // ensure const-correctness of operator()
  }

  SECTION("make_cmap<double> makes read-only TensorMap from double*") {
    auto map = make_cmap(ptr0, Range{2, 3, 4});
    auto map_elemaccess_returns_constref = std::is_same<decltype(map(0,0,0)),const double&>::value;
    CHECK(map_elemaccess_returns_constref); // ensure const-correctness of operator()
  }

  SECTION("make_cmap<double> makes read-only TensorMap from const double*") {
    auto map = make_cmap(cptr0, Range{2, 3, 4});
    auto map_elemaccess_returns_constref = std::is_same<decltype(map(0,0,0)),const double&>::value;
    CHECK(map_elemaccess_returns_constref); // ensure const-correctness of operator()
  }

} // TEST_CASE("TensorView constructors")

// serialization of TensorView is disabled
#if 0
TEST_CASE("TensorView serialization") {
  SECTION("TensorView<double>") {
    const auto archive_fname = "tensorview_serialziation.archive";

    DTensor T0(2, 3, 4);
    fillEls(T0);
    auto T0v = make_view(T0);
    // write
    {
      std::ofstream os(archive_fname);
      assert(os.good());
      boost::archive::xml_oarchive ar(os);
      ar << BOOST_SERIALIZATION_NVP(T0v);
    }
    // read
    {
      std::ifstream is(archive_fname);
      assert(is.good());
      boost::archive::xml_iarchive ar(is);

      decltype(T0v) T0v_copy;
      ar >> BOOST_SERIALIZATION_NVP(T0v_copy);

      CHECK(T0v == T0v_copy);
    }
    std::remove(archive_fname);
  }
}
#endif
