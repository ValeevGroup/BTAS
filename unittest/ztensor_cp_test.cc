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

TEST_CASE("ZCP")
{
  typedef btas::Tensor<double> tensor;
  typedef btas::Tensor<std::complex<double>> ztensor;
  using zconv_class = btas::FitCheck<ztensor>;
  //using conv_class = btas::FitCheck<tensor>;
  using conv_class_coupled = btas::CoupledFitCheck<tensor>;
  using btas::CP_ALS;
  using btas::CP_RALS;
  using btas::CP_DF_ALS;
  using btas::COUPLED_CP_ALS;

  //double epsilon = fmax(1e-10, std::numeric_limits<double>::epsilon());
  double epsilon = 1e-7;

  ztensor Z4(4,2,7,3);
  std::ifstream inp (__dirname + "/z-mat4D.txt");
  if (inp.is_open()){
    int i,j,k,l;
    double rel,img;
    while(inp){
      inp >> i >> j >> k>>l>> rel>> img;
      std::complex<double> val (rel,img);
      Z4(i,j,k,l) = val;
    }
  }


  ztensor Z44(Z4.extent(1), Z4.extent(2), Z4.extent(3), Z4.extent(1), Z4.extent(2), Z4.extent(3));
  std::complex<double> one {1.0,0.0};
  std::complex<double> zero{0.0,0.0};
  contract(one, Z4, {1,2,3,4}, Z4, {1,5,6,7}, zero, Z44, {2,3,4,5,6,7});
  std::complex<double> norm4 = sqrt(dot(Z4, Z4));
  std::complex<double> norm42 = sqrt(dot(Z44, Z44));

  zconv_class conv(1e-3);

  // ALS tests
    SECTION("ALS MODE = 4, Finite error"){
      CP_ALS<ztensor, zconv_class> A1(Z4);
      conv.set_norm(norm4.real());
      double diff = 1.0 - A1.compute_error(conv, 1e-2, 1, 99);
      std::cout << diff << std::endl;
    }
}
#endif
