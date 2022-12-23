#ifdef BTAS_HAS_BLAS_LAPACK
#include <btas/btas.h>
#include <btas/generic/converge_class.h>
#include "../unittest/test.h"
#include <fstream>
#include <iomanip>
#include <iostream>

#include <libgen.h>

#define BTAS_ENABLE_TUCKER_CP_UT 1
#define BTAS_ENABLE_RANDOM_CP_UT 0

const std::string __dirname = dirname(strdup(__FILE__));

TEST_CASE("CP")
{
  typedef btas::Tensor<double> tensor;
  typedef btas::Tensor<std::complex<double>> ztensor;
  using conv_class = btas::FitCheck<tensor>;
  using conv_class_coupled = btas::CoupledFitCheck<tensor>;
  using btas::CP_ALS;
  using btas::CP_RALS;
  using btas::CP_DF_ALS;
  using btas::COUPLED_CP_ALS;

  //double epsilon = fmax(1e-10, std::numeric_limits<double>::epsilon());
  double epsilon = 1e-7;
  // TEST_CASE("CP_ALS"){

  ztensor Z4(4,2,7,3);
  auto n1 = Z4.extent(0);
  std::ifstream inp ("4D_complex.txt");
  if (inp.is_open()){
    int i,j,k,l;
    double rel,img;
    while(inp){
      inp >> i >> j >> k>>l>> rel>> img;
      Z4(i,j,k,l) = {rel,img};
    }
  }

  tensor D4(6, 3, 3, 7);
  std::ifstream in4;
  in4.open(__dirname + "/mat4D.txt");
  CHECK(in4.is_open());
  for (auto &i : D4) {
    in4 >> i;
  }
  in4.close();

  tensor results(40, 1);
  std::ifstream res(__dirname + "/cp_test_results.txt");
  CHECK(res.is_open());
  for (auto &i : results) {
    res >> i;
  }

  ztensor Z44(Z4.extent(1), Z4.extent(2), Z4.extent(3), Z4.extent(1), Z4.extent(2), Z4.extent(3));

  std::complex<double> one {1.0,0.0};
  std::complex<double> zero{0.0,0.0};
  contract(one, Z4, {1,2,3,4}, Z4, {1,5,6,7}, zero, Z44, {2,3,4,5,6,7});

  std::complex<double> norm4 = sqrt(dot(Z4, Z4));

  std::complex<double> norm42 = sqrt(dot(Z44, Z44));


  conv_class conv(1e-3);

  // ALS tests
    SECTION("ALS MODE = 4, Finite error"){
      CP_ALS<tensor, conv_class> A1(Z4);
      conv.set_norm(norm4);
      double diff = 1.0 - A1.compute_error(conv, 1e-2, 1, 99);
      CHECK(std::abs(diff - results(5,0)) <= epsilon);
    }
}
#endif
