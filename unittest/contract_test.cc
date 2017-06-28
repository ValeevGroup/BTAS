#include "test.h"

#include <iostream>
#include <random>

#include "btas/generic/contract.h"
#include "btas/tensor.h"

using std::cout;
using std::endl;

using btas::Range;

using DTensor = btas::Tensor<double>;

static std::ostream& operator<<(std::ostream& s, const DTensor& X) {
  for (const auto& i : X.range()) {
    btas::array_adaptor<typename DTensor::index_type>::print(i, s);
    s << " " << X(i) << "\n";
  }
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

// rng() is useful for filling Tensors with random elements
// using the code T.generate(rng);
double static rng() {
  static std::mt19937 rng(std::time(NULL));
  static auto dist = std::uniform_real_distribution<double>{0., 1.};
  return dist(rng);
}

TEST_CASE("Tensor Contract") {
  DTensor T2(3, 2);
  fillEls(T2);

  DTensor T3(3, 2, 4);
  fillEls(T3);

  SECTION("Matrix-like") {
    // cout << "T2 = \n" << T2 << endl;
    enum { i, j, k };

    DTensor A;
    contract(1.0, T2, {j, i}, T2, {j, k}, 0.0, A, {i, k});

    auto rmax = T2.extent(1), cmax = T2.extent(1);
    for (size_t r = 0; r < rmax; ++r)
      for (size_t c = 0; c < cmax; ++c) {
        double val = 0;
        for (size_t i = 0; i < T2.extent(0); ++i) {
          val += T2(i, r) * T2(i, c);
        }
        //           CHECK(val == A(r,c));
        // cout << r << " " << c << " " << val << " " << R(r,c) << endl;
      }

    DTensor B;
    contract(1.0, T2, {i, j}, T2, {k, j}, 0.0, B, {i, k});
    // cout << "B = \n" << B << endl;

    rmax = T2.extent(0), cmax = T2.extent(0);
    for (size_t r = 0; r < rmax; ++r)
      for (size_t c = 0; c < cmax; ++c) {
        double val = 0;
        for (size_t n = 0; n < T2.extent(1); ++n) {
          val += T2(r, n) * T2(c, n);
        }
        //            CHECK(val == B(r,c));
        // cout << r << " " << c << " " << val << " " << B(r,c) << endl;
      }
  }

  SECTION("Memory Bug #56") {
    //
    // Regression test for github issue #56
    //
    DTensor T(3, 4);
    T.generate(rng);
    enum { i, j, k };
    DTensor R;
    contract(1.0, T, {j, i}, T, {j, k}, 0.0, R, {i, k});
  }
}
