//#ifdef BTAS_HAS_CBLAS
//#include "test.h"

#include "btas/btas.h"
#include "btas/generic/converge_class.h"
#include "test.h"

#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char *argv[]) {
  typedef btas::Tensor<double> tensor;
  using conv_class = btas::FitCheck<tensor>;
  using conv_class_coupled = btas::CoupledFitCheck<tensor>;

  double epsilon = fmax(1e-10, std::numeric_limits<double>::epsilon());
  // TEST_CASE("CP_ALS"){
  tensor D3(5, 2, 9);
  std::ifstream in3;
  in3.open("./mat3D.txt");
  if (!in3.is_open()) {
    std::cout << "File isn't open" << std::endl;
    return 1;
  }
  for (auto &i : D3) {
    in3 >> i;
  }
  in3.close();

  tensor D4(6, 3, 3, 7);
  std::ifstream in4;
  in4.open("./mat4D.txt");
  if (!in4.is_open()) {
    std::cout << "File isn't open" << std::endl;
    return 1;
  }
  for (auto &i : D4) {
    in4 >> i;
  }
  in4.close();

  tensor D5(2, 6, 1, 9, 3);
  std::ifstream in5;
  in5.open("./mat5D.txt");
  if (!in5.is_open()) {
    std::cout << "File isn't open" << std::endl;
    return 1;
  }
  for (auto &i : D5) {
    in5 >> i;
  }
  in5.close();

  tensor results(36, 1);
  std::ifstream res("./cp_test_results.txt", std::ifstream::in);
  if (!res.is_open()) {
    std::cout << "Results are not open " << std::endl;
    // return;
  }
  for (auto &i : results) {
    res >> i;
  }
  tensor D33(D3.extent(1), D3.extent(2), D3.extent(1), D3.extent(2));
  tensor D44(D4.extent(1), D4.extent(2), D4.extent(3), D4.extent(1), D4.extent(2), D4.extent(3));
  tensor D55(D5.extent(1), D5.extent(2), D5.extent(3), D5.extent(4), D5.extent(1), D5.extent(2), D5.extent(3), D5.extent(4));

  contract(1.0, D3, {1,2,3}, D3, {1,4,5}, 0.0, D33, {2,3,4,5});
  contract(1.0, D4, {1,2,3,4}, D4, {1,5,6,7}, 0.0, D44, {2,3,4,5,6,7});
  contract(1.0, D5, {1,2,3,4,5}, D5, {1,6,7,8,9}, 0.0, D55, {2,3,4,5,6,7,8,9});

  double norm3 = sqrt(dot(D3, D3));
  double norm4 = sqrt(dot(D4, D4));
  double norm5 = sqrt(dot(D5, D5));

  double norm32 = sqrt(dot(D33, D33));
  double norm42 = sqrt(dot(D44, D44));
  double norm52 = sqrt(dot(D55, D55));

  conv_class conv(1e-3);

  // ALS tests
  TEST_CASE("CP-ALS")
  {
    SECTION("ALS MODE = 3, Finite rank"){
      CP_ALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff =
              A1.compute_rank(5, conv, 1, false, 0, 100, true, false, true);
      std::cout << std::setprecision(16) << diff << std::endl;
      CHECK((diff - results(0,0)) <= epsilon);
    }
    SECTION("ALS MODE = 3, Finite error"){
      CP_ALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff = A1.compute_error(conv, 1e-2, 1, 100);
      std::cout << std::setprecision(16) << diff << std::endl;
      CHECK((diff - results(1,0)) <= epsilon);
    }
#ifdef _HAS_INTEL_MKL
    SECTION("ALS MODE = 3, Tucker + CP"){
      auto d = D3;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff =
              A1.compress_compute_tucker(0.1, conv, false, 1e-2, 5, true, false);
      CHECK((diff - results(2,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
    SECTION("ALS MODE = 3, Random + CP"){
      auto d = D3;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff =
              A1.compress_compute_rand(2, conv, 0, 2, false, 1e-2, 5);
      CHECK((diff - results(3,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
#endif

    SECTION("ALS MODE = 4, Finite rank"){
      CP_ALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff =
              A1.compute_rank(5, conv);
      CHECK((diff - results(4,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
    SECTION("ALS MODE = 4, Finite error"){
      CP_ALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff = A1.compute_error(conv, 1e-2, 1, 100);
      CHECK((diff - results(5,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
#ifdef _HAS_INTEL_MKL
    SECTION("ALS MODE = 4, Tucker + CP"){
      auto d = D4;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = A1.compress_compute_tucker(0.1, conv, false, 1e-2, 5);
      CHECK((diff - results(6,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
    SECTION("ALS MODE = 4, Random + CP"){
      auto d = D4;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = A1.compress_compute_rand(3, conv, 0, 2, true, 1e-2, 0, true, false, 1, 20, 100);
      std::cout << std::setprecision(16) << diff << std::endl;
      CHECK((diff - results(7,0)) <= epsilon);
    }
#endif

    SECTION("ALS MODE = 5, Finite rank"){
      CP_ALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_rank(5, conv);
      CHECK((diff - results(8,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
     SECTION("ALS MODE = 5, Finite error"){
      CP_ALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_error(conv, 1e-2, 1, 20);
      CHECK((diff - results(9,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
#ifdef _HAS_INTEL_MKL
    SECTION("ALS MODE = 5, Tucker + CP"){
      auto d = D5;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = A1.compress_compute_tucker(0.1, conv, false, 1e-2, 5, true, false);
      CHECK((diff - results(10,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
    SECTION("ALS MODE = 5, Random + CP"){
      auto d = D5;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = A1.compress_compute_rand(1, conv, 0, 2, false, 1e-2, 5);
      CHECK((diff - results(11,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
#endif
  }
  //std::cout << "done with ALS" << std::endl;
  // RALS tests
  TEST_CASE("CP-RALS")
  {
    SECTION("RALS MODE = 3, Finite rank"){
      CP_RALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff =
              A1.compute_rank(5, conv, 1, false, 0, 100, true, false, true);
      std::cout << std::setprecision(16) << diff << std::endl;
      CHECK((diff - results(12,0)) <= epsilon);
    }
    SECTION("RALS MODE = 3, Finite error"){
      CP_RALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff = A1.compute_error(conv, 1e-2, 1, 100);
      std::cout << std::setprecision(16) << diff << std::endl;
      CHECK((diff - results(13,0)) <= epsilon);
    }
#ifdef _HAS_INTEL_MKL
    SECTION("RALS MODE = 3, Tucker + CP"){
      auto d = D3;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff =
              A1.compress_compute_tucker(0.1, conv, false, 1e-2, 5, true, false);
      CHECK((diff - results(14,0)) <= epsilon);
      std::cout << std::setprecision(16) << diff << std::endl;
    }
    SECTION("RALS MODE = 3, Random + CP"){
      auto d = D3;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff =
              A1.compress_compute_rand(2, conv, 0, 2, false, 1e-2, 5);
      CHECK((diff - results(15,0)) <= epsilon);
    }
#endif

    SECTION("RALS MODE = 4, Finite rank"){
      CP_RALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff =
              A1.compute_rank(5, conv);
      CHECK((diff - results(16,0)) <= epsilon);
    }
    SECTION("RALS MODE = 4, Finite error"){
      CP_RALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff = A1.compute_error(conv, 1e-2, 1, 100);
      CHECK((diff - results(17,0)) <= epsilon);
    }
#ifdef _HAS_INTEL_MKL
    SECTION("RALS MODE = 4, Tucker + CP"){
      auto d = D4;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = A1.compress_compute_tucker(0.1, conv, false, 1e-2, 5);
      CHECK((diff - results(18,0)) <= epsilon);
    }
    SECTION("RALS MODE = 4, Random + CP"){
      auto d = D4;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = A1.compress_compute_rand(3, conv, 0, 2, true, 1e-2, 0, true, false, 1, 20, 100);
      CHECK((diff - results(19,0)) <= epsilon);
    }
#endif

    SECTION("RALS MODE = 5, Finite rank"){
      CP_RALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_rank(5, conv);
      CHECK((diff - results(20,0)) <= epsilon);
    }
    SECTION("RALS MODE = 5, Finite error"){
      CP_RALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_error(conv, 1e-2, 1, 20);
      CHECK((diff - results(21,0)) <= epsilon);
    }
#ifdef _HAS_INTEL_MKL
    SECTION("RALS MODE = 5, Tucker + CP"){
      auto d = D5;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = A1.compress_compute_tucker(0.1, conv, false, 1e-2, 5, true, false);
      CHECK((diff - results(22,0)) <= epsilon);
    }
    SECTION("RALS MODE = 5, Random + CP"){
      auto d = D5;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = A1.compress_compute_rand(1, conv, 0, 2, false, 1e-2, 5);
      CHECK((diff - results(23,0)) <= epsilon);
    }
#endif
  }
  //std::cout << "done with RALS" << std::endl;
  // CP-DF-ALS tests
  TEST_CASE("CP-DF-ALS")
  {
    SECTION("DF-ALS MODE = 3, Finite rank"){
      CP_DF_ALS<tensor, conv_class> A1(D3, D3);
      conv.set_norm(norm32);
      double diff = A1.compute_rank(5, conv);
      CHECK((diff - results(24,0)) <= epsilon);
    }
     SECTION("DF-ALS MODE = 3, Finite error"){
      CP_DF_ALS<tensor, conv_class> A1(D3, D3);
      conv.set_norm(norm32);
      double diff = A1.compute_error(conv, 1e-2, 1, 20);
      CHECK((diff - results(25,0)) <= epsilon);
    }
     SECTION("DF-ALS MODE = 4, Finite rank"){
      CP_DF_ALS<tensor,conv_class> A1(D4,D4);
      conv.set_norm(norm42);
      double diff = A1.compute_rank(5,conv);
      CHECK((diff - results(26,0)) <= epsilon);
    }
    SECTION("DF-ALS MODE = 4, Finite error"){
      CP_DF_ALS<tensor,conv_class>A1(D4,D4);
      conv.set_norm(norm42);
      double diff = A1.compute_error(conv, 1e-2, 1, 20);
      CHECK((diff - results(27,0)) <= epsilon);
    }
    SECTION("DF-ALS MODE = 5, Finite rank"){
      CP_DF_ALS<tensor,conv_class> A1(D5,D5);
      conv.set_norm(norm52);
      double diff = A1.compute_rank(5,conv);
      CHECK((diff - results(28,0)) <= epsilon);
    }
    SECTION("DF-ALS MODE = 5, Finite error"){
      CP_DF_ALS<tensor,conv_class>A1(D5,D5);
      conv.set_norm(norm52);
      double diff = A1.compute_error(conv, 1e-2, 1, 20);
      CHECK((diff - results(29,0)) <= epsilon);
    }
  }
  //std::cout << "done with DF ALS" << std::endl;
  // coupled ALS test
  TEST_CASE("COUPLED-CP-ALS")
  {
    SECTION("COUPLED-ALS MODE = 3, Finite rank"){
      CoupledFitCheck<tensor> conv_coupled(3, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D3, D3);
      conv_coupled.set_norm(norm3, norm3);
      double diff = A1.compute_rank(5, conv_coupled);
      CHECK((diff - results(30,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 3, Finite error"){
      CoupledFitCheck<tensor> conv_coupled(3, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D3, D3);
      conv_coupled.set_norm(norm3, norm3);
      double diff = A1.compute_error(conv_coupled, 1e-2, 1, 20);
      CHECK((diff - results(31,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 4, Finite rank"){
      CoupledFitCheck<tensor> conv_coupled(4, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D4, D4);
      conv_coupled.set_norm(norm4, norm4);
      double diff = A1.compute_rank(5,conv_coupled);
      CHECK((diff - results(32,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 4, Finite error"){
      CoupledFitCheck<tensor> conv_coupled(4, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D4, D4);
      conv_coupled.set_norm(norm4, norm4);
      double diff = A1.compute_error(conv_coupled, 1e-2, 1, 20);
      CHECK((diff - results(33,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 5, Finite rank"){
      CoupledFitCheck<tensor> conv_coupled(5, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D5, D5);
      conv_coupled.set_norm(norm5, norm5);
      double diff = A1.compute_rank(5,conv_coupled);
      CHECK((diff - results(34,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 5, Finite error"){
      CoupledFitCheck<tensor> conv_coupled(5, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D5, D5);
      conv_coupled.set_norm(norm5, norm5);
      double diff = A1.compute_error(conv_coupled, 1e-2, 1, 20);
      CHECK((diff - results(35,0)) <= epsilon);
    }
  }
  //std::cout << "done with coupled ALS" << std::endl;
}
//#endif //BTAS_HAS_CBLAS
