#include "btas/tensor.h"
#include <btas/btas.h>
#include <btas/tarray.h>
#include <random>
#include "btas/tarray.h"
#include "btas/tensorview.h"
#include "test.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>

#ifdef BTAS_HAS_BOOST_SERIALIZATION
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/complex.hpp>
#endif  // BTAS_HAS_BOOST_SERIALIZATION

using std::cout;
using std::endl;

using btas::Range;
using btas::Tensor;

using DTensor = Tensor<double>;
using namespace btas;
using namespace std;

double static rng() {
  static std::mt19937 rng(std::time(NULL));
  static auto dist = std::uniform_real_distribution<double>{0., 1.};
  return dist(rng);
}

static std::ostream& operator<<(std::ostream& s, const DTensor& X) {
  for (auto i : X.range()) s << i << " " << X(i) << "\n";
  return s;
}

// Set the elements of a Tensor T such that
// T(i,j,k) = 1ijk
// assuming individual dimensions are all less than 10.
// (The 1 in front is just a placeholder.)
void static fillEls(DTensor& T) {
  if (T.rank() == 0) return;
  const double base = pow(10., T.rank());
  const size_t max_ii = T.rank() - 1;
  for (auto I : T.range()) {
    double& val = T(I);
    val = base;
    for (size_t ii = 0; ii <= max_ii; ++ii) {
      val += I[ii] * pow(10., max_ii - ii);
    }
  }
}

template <typename T>
T randomReal() {
  static std::mt19937 rng(std::time(NULL));
  static auto dist = std::uniform_real_distribution<T>{0., 1.};
  return dist(rng);
}

TEST_CASE("Tensor Constructors") {
  SECTION("Default Constructor") {
    Tensor<double> T0;
    CHECK(T0.size() == 0);
    CHECK(T0.empty());
    Tensor<int> T1;
    CHECK(T1.size() == 0);
    CHECK(T1.empty());
    Tensor<bool> T2;
    CHECK(T2.size() == 0);
    CHECK(T2.empty());
    Tensor<std::complex<double>> T3;
    CHECK(T3.size() == 0);
    CHECK(T3.empty());
  }

  SECTION("Extent Constructor") {
    DTensor T0(2, 3, 4);
    CHECK(T0.rank() == 3);
    CHECK(T0.extent(0) == 2);
    CHECK(T0.extent(1) == 3);
    CHECK(T0.extent(2) == 4);
    CHECK(T0.size() == 2 * 3 * 4);
    CHECK(!T0.empty());

    DTensor T1(6, 1, 2, 7, 9);
    CHECK(T1.rank() == 5);
    CHECK(T1.extent(0) == 6);
    CHECK(T1.extent(1) == 1);
    CHECK(T1.extent(2) == 2);
    CHECK(T1.extent(3) == 7);
    CHECK(T1.extent(4) == 9);
    CHECK(T1.size() == 6 * 1 * 2 * 7 * 9);
    CHECK(!T1.empty());

    Tensor<int> Ti(2, 4, 5);
    CHECK(Ti.rank() == 3);
    CHECK(Ti.extent(0) == 2);
    CHECK(Ti.extent(1) == 4);
    CHECK(Ti.extent(2) == 5);
    CHECK(Ti.size() == 2 * 4 * 5);

    Tensor<std::complex<double>> Tc(3, 2, 9);
    CHECK(Tc.rank() == 3);
    CHECK(Tc.extent(0) == 3);
    CHECK(Tc.extent(1) == 2);
    CHECK(Tc.extent(2) == 9);
    CHECK(Tc.size() == 3 * 2 * 9);
  }

  SECTION("Range Constructor") {
    Range r0(2, 5, 3);
    DTensor T0(r0);
    CHECK(T0.rank() == 3);
    CHECK(T0.extent(0) == 2);
    CHECK(T0.extent(1) == 5);
    CHECK(T0.extent(2) == 3);

    Range r1(2, 5, 3, 9, 18, 6);
    DTensor T1(r1);
    CHECK(T1.rank() == 6);
  }
  SECTION("Fixed Rank Tensor") {
    TArray<double, 3> T0(2, 4, 3);
    CHECK(T0.rank() == 3);
    CHECK(T0.size() == 24);
    CHECK(T0.extent(0) == 2);
    CHECK(T0.extent(1) == 4);
    CHECK(T0.extent(2) == 3);

    Range r1(2, 5, 3);
    TArray<double, 3> T1(r1);

    typedef TArray<double, 3>::range_type Range3;  // rank-3 Range
    Range3 r2(2, 5, 3);
    TArray<double, 3> T2(r2);

    CHECK(T1 == T2);

    typedef TArray<double, 4>::range_type Range4;  // rank-3 Range
    Range4 r3(2, 5, 3, 4);
    // TArray<double,3> T3(r3); // error: mismatched rank
  }
}

TEST_CASE("Custom Tensor") {
  SECTION("Storage") {
    {
      typedef Tensor<double, btas::DEFAULT::range, std::vector<double>> Tensor;
      Tensor T0;
      Tensor T1(2, 3, 4);
    }
    {
      typedef Tensor<double, btas::DEFAULT::range, btas::varray<double>> Tensor;
      Tensor T0;
      Tensor T1(2, 3, 4);
    }
    {
      typedef Tensor<double, btas::DEFAULT::range, std::array<double, 24>>
          Tensor;
      Tensor T0;
      Tensor T1(2, 3, 4);
    }
    {
      typedef Tensor<double, btas::DEFAULT::range, std::valarray<double>>
          Tensor;
      Tensor T0;
      Tensor T1(2, 3, 4);
    }
  }
}

TEST_CASE("Tensor Operations") {
  DTensor T2(3, 2);
  fillEls(T2);

  DTensor T3(3, 2, 4);
  fillEls(T3);

  SECTION("Fill") {
    T3.fill(1.);
    for (auto x : T3) CHECK(x == 1);
  }

  SECTION("Element Access") {
    CHECK(T3(2,1,3) == 1213);
  }

  SECTION("Generate") {
    std::vector<double> data(T3.size());
    for (auto& x : data) x = rng();

    size_t count = 0;
    T3.generate([&]() { return data[count++]; });

    auto it = T3.cbegin();
    size_t j = 0;
    for (; it != T3.cend(); ++j, ++it) {
      CHECK(*it == data[j]);
    }
  }

  SECTION("Tensor of Tensor") {
    Tensor<Tensor<double>> A(4, 3);
    Tensor<double> aval(2, 3);
    aval.generate([]() { return randomReal<double>(); });
    A.fill(aval);
    Tensor<Tensor<double>> B(3, 2);
    Tensor<double> bval(3, 4);
    bval.generate([]() { return randomReal<double>(); });
    B.fill(bval);
    Tensor<Tensor<double>> C(4, 2);
    C.fill(Tensor<double>(
        2, 4));  // rank info is required to determine contraction ranks at gemm
    btas::gemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 1.0, C);
    Tensor<Tensor<double>> Ctest(4, 2);
    Ctest.fill(Tensor<double>(2, 4));
    for (size_t i0 = 0; i0 < A.extent(0); i0++)
      for (size_t i1 = 0; i1 < A.extent(1); i1++)
        for (size_t i2 = 0; i2 < B.extent(1); i2++)
          btas::gemm(CblasNoTrans, CblasNoTrans, 1.0, A(i0, i1), B(i1, i2), 1.0,
                     Ctest(i0, i2));

    const auto eps_double = 1.e4 * std::numeric_limits<double>::epsilon();
    Ctest -= C;
    CHECK(dot(Ctest, Ctest) < eps_double);

    Tensor<Tensor<double>> Ctest1(4, 2);
    Ctest1.fill(Tensor<double>(2, 4));
    for (size_t i0 = 0; i0 < A.extent(0); i0++)
      for (size_t i1 = 0; i1 < A.extent(1); i1++)
        for (size_t i2 = 0; i2 < B.extent(1); i2++)
          for (size_t j0 = 0; j0 < A(i0, i1).extent(0); j0++)
            for (size_t j1 = 0; j1 < A(i0, i1).extent(1); j1++)
              for (size_t j2 = 0; j2 < B(i1, i2).extent(1); j2++)
                Ctest1(i0, i2)(j0, j2) += A(i0, i1)(j0, j1) * B(i1, i2)(j1, j2);
    Ctest1 -= C;
    CHECK(dot(Ctest1, Ctest1) < eps_double);
  }

#ifdef BTAS_HAS_BOOST_SERIALIZATION
  SECTION("Serialization") {
    const auto archive_fname = "tensor_operations.serialization.archive";

    Tensor<std::array<complex<double>, 3>> T1(2, 3, 4);
    T1.fill({{{1.0, 2.0}, {2.0, 1.0}, {2.0, 3.0}}});
    // Tensor<double,3> t({-1,0,1},{2,4,3});
    TArray<double, 3> t(2, 4, 3);
    t.generate([]() { return randomReal<double>(); });
    Tensor<Tensor<double>> A(4, 4);
    Tensor<double> aval(2, 2);
    aval.fill(1.0);
    A.fill(aval);
    // write
    {
      std::ofstream os(archive_fname);
      assert(os.good());
      boost::archive::xml_oarchive ar(os);
      CHECK_NOTHROW(ar << BOOST_SERIALIZATION_NVP(t));   // fixed-size Tensor
      CHECK_NOTHROW(ar << BOOST_SERIALIZATION_NVP(A));   // Tensor of Tensor
      CHECK_NOTHROW(ar << BOOST_SERIALIZATION_NVP(T1));  // Tensor of complex datatypes
    }
    // read
    {
      std::ifstream is(archive_fname);
      assert(is.good());
      boost::archive::xml_iarchive ar(is);

      TArray<double, 3> tcopy;
      CHECK_NOTHROW(ar >> BOOST_SERIALIZATION_NVP(tcopy));
      CHECK(t == tcopy);

      Tensor<Tensor<double>> Acopy;
      CHECK_NOTHROW(ar >> BOOST_SERIALIZATION_NVP(Acopy));
      CHECK(A == Acopy);

      Tensor<std::array<complex<double>, 3>> T1copy;
      CHECK_NOTHROW(ar >> BOOST_SERIALIZATION_NVP(T1copy));
      CHECK(T1 == T1copy);
    }
    std::remove(archive_fname);
  }
#endif  // BTAS_HAS_BOOST_SERIALIZATION

}
