#ifdef BTAS_HAS_BLAS_LAPACK
#include <btas/btas.h>
#include <btas/generic/converge_class.h>
#include <libgen.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "../unittest/test.h"

#define BTAS_ENABLE_TUCKER_CP_UT 1
#define BTAS_ENABLE_RANDOM_CP_UT 0

const std::string __dirname = dirname(strdup(__FILE__));

TEST_CASE("ZCP") {
  typedef btas::Tensor<double> tensor;
  typedef btas::Tensor<std::complex<double>> ztensor;
  using zconv_class = btas::FitCheck<ztensor>;
  // using conv_class = btas::FitCheck<tensor>;
  using conv_class_coupled = btas::CoupledFitCheck<tensor>;
  using btas::COUPLED_CP_ALS;
  using btas::CP_ALS;
  using btas::CP_DF_ALS;
  using btas::CP_RALS;

  // double epsilon = fmax(1e-10, std::numeric_limits<double>::epsilon());
  double epsilon = 1e-5;

  ztensor Z3(3, 2, 4);
  std::ifstream inp3(__dirname + "/z-mat3D.txt");
  if (inp3.is_open()) {
    int i, j, k;
    double rel, img;
    while (inp3) {
      inp3 >> i >> j >> k >> rel >> img;
      std::complex<double> val(rel, img);
      Z3(i, j, k) = val;
    }
  }

  ztensor Z4(4, 2, 7, 3);
  std::ifstream inp4(__dirname + "/z-mat4D.txt");
  if (inp4.is_open()) {
    int i, j, k, l;
    double rel, img;
    while (inp4) {
      inp4 >> i >> j >> k >> l >> rel >> img;
      std::complex<double> val(rel, img);
      Z4(i, j, k, l) = val;
    }
  }

  tensor results(43, 1);
  std::ifstream res(__dirname + "/cp_test_results.txt");
  CHECK(res.is_open());
  for (auto &i : results) {
    res >> i;
  }

  ztensor Z44(Z4.extent(1), Z4.extent(2), Z4.extent(3), Z4.extent(1), Z4.extent(2), Z4.extent(3));
  std::complex<double> one{1.0, 0.0};
  std::complex<double> zero{0.0, 0.0};
  contract(one, Z4, {1, 2, 3, 4}, Z4, {1, 5, 6, 7}, zero, Z44, {2, 3, 4, 5, 6, 7});
  std::complex<double> norm4 = sqrt(dot(Z4, Z4));
  std::complex<double> norm42 = sqrt(dot(Z44, Z44));
  std::complex<double> norm3 = sqrt(dot(Z3, Z3));

  zconv_class conv(1e-3);

  // ALS tests
  SECTION("ALS MODE = 3, Finite error") {
    CP_ALS<ztensor, zconv_class> A1(Z3);
    conv.set_norm(norm3.real());
    double diff = A1.compute_error(conv, 1e-9, 1, 50, false, 0, 1e4, false, true);
    CHECK(std::abs(diff) <= epsilon);
  }
  SECTION("ALS MODE = 3, Finite rank") {
    CP_ALS<ztensor, zconv_class> A1(Z4);
    conv.set_norm(norm4.real());
    double diff = A1.compute_rank(99, conv);
    CHECK(std::abs(diff) <= epsilon);
  }

  SECTION("ALS MODE = 4, Finite error") {
    CP_ALS<ztensor, zconv_class> A1(Z4);
    conv.set_norm(norm4.real());
    double diff = A1.compute_error(conv, 1e-9, 1, 99, false, 0, 1e4, false, true);
    CHECK(std::abs(diff) <= epsilon);
  }
  SECTION("ALS MODE = 4, Finite rank") {
    CP_ALS<ztensor, zconv_class> A1(Z4);
    conv.set_norm(norm4.real());
    double diff = A1.compute_rank(99, conv);
    CHECK(std::abs(diff) <= epsilon);
  }
}
#endif
