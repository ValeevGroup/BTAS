#include <btas/btas.h>
#include <btas/generic/converge_class.h>
#include "../unittest/test.h"

#include <fstream>
#include <iomanip>
#include <iostream>

#include <libgen.h>

const std::string __dirname = dirname(strdup(__FILE__));

TEST_CASE("CP")
{
  typedef btas::Tensor<double> tensor;
  using conv_class = btas::FitCheck<tensor>;
  using conv_class_coupled = btas::CoupledFitCheck<tensor>;
  using btas::CP_ALS;
  using btas::CP_RALS;
  using btas::CP_DF_ALS;
  using btas::COUPLED_CP_ALS;

  //double epsilon = fmax(1e-10, std::numeric_limits<double>::epsilon());
  double epsilon = 1e-7;
  // TEST_CASE("CP_ALS"){
  tensor D3(5, 2, 9);
  std::ifstream in3(__dirname + "/mat3D.txt");
  CHECK(in3.is_open());
  for (auto &i : D3) {
    in3 >> i;
  }
  in3.close();

  tensor D4(6, 3, 3, 7);
  std::ifstream in4;
  in4.open(__dirname + "/mat4D.txt");
  CHECK(in4.is_open());
  for (auto &i : D4) {
    in4 >> i;
  }
  in4.close();

  tensor D5(2, 6, 1, 9, 3);
  std::ifstream in5;
  in5.open(__dirname + "/mat5D.txt");
  CHECK(in5.is_open());
  for (auto &i : D5) {
    in5 >> i;
  }
  in5.close();

  tensor results(36, 1);
  std::ifstream res(__dirname + "/cp_test_results.txt");
  CHECK(res.is_open());
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
  {
    SECTION("ALS MODE = 3, Finite rank"){
      CP_ALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff =
              A1.compute_rank(5, conv, 1, false, 0, 100, false, false, true);
      CHECK((diff - results(0,0)) <= epsilon);
    }
    SECTION("ALS MODE = 3, Finite error"){
      CP_ALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff = A1.compute_error(conv, 1e-2, 1, 100);
      CHECK((diff - results(1,0)) <= epsilon);
    }
    SECTION("ALS MODE = 3, Tucker + CP"){
      auto d = D3;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff =
              A1.compress_compute_tucker(0.1, conv, 5, true, false, 100, true);
      CHECK((diff - results(2,0)) <= epsilon);
    }
    SECTION("ALS MODE = 3, Random + CP"){
      auto d = D3;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff =
              A1.compress_compute_rand(2, conv, 0, 2, 5, true, false, 100, true);
      CHECK((diff - results(3,0)) <= epsilon);
    }

    SECTION("ALS MODE = 4, Finite rank"){
      CP_ALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff =
              A1.compute_rank(5, conv);
      CHECK((diff - results(4,0)) <= epsilon);
    }
    SECTION("ALS MODE = 4, Finite error"){
      CP_ALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff = A1.compute_error(conv, 1e-2, 1, 100);
      CHECK((diff - results(5,0)) <= epsilon);
    }
    SECTION("ALS MODE = 4, Tucker + CP"){
      auto d = D4;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = A1.compress_compute_tucker(0.1, conv, 5, true, false, 1e4, true);
      CHECK((diff - results(6,0)) <= epsilon);
    }
    SECTION("ALS MODE = 4, Random + CP"){
      auto d = D4;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = A1.compress_compute_rand(3, conv, 0, 2, 1, true, false, 20, true);
      CHECK((diff - results(7,0)) <= epsilon);
    }

    SECTION("ALS MODE = 5, Finite rank"){
      CP_ALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_rank(5, conv);
      CHECK((diff - results(8,0)) <= epsilon);
    }
     SECTION("ALS MODE = 5, Finite error"){
      CP_ALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_error(conv, 1e-2, 10, 200);
      CHECK((diff - results(9,0)) <= epsilon);
    }
    SECTION("ALS MODE = 5, Tucker + CP"){
      auto d = D5;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = A1.compress_compute_tucker(0.1, conv, 5, true, false, 100, true);
      CHECK((diff - results(10,0)) <= epsilon);
    }
    SECTION("ALS MODE = 5, Random + CP"){
      auto d = D5;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = A1. compress_compute_rand(1, conv, 0, 2, 5, true, false, 100, false);
      CHECK((diff - results(11,0)) <= epsilon);
    }
      
  }
#if 0
  // RALS tests
  {
    SECTION("RALS MODE = 3, Finite rank"){
      CP_RALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff =
              A1.compute_rank(5, conv, 1, false, 0, 100, false, false, true);
      CHECK((diff - results(12,0)) <= epsilon);
    }
    SECTION("RALS MODE = 3, Finite error"){
      CP_RALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff = A1.compute_error(conv, 1e-2, 1, 100);
      CHECK((diff - results(13,0)) <= epsilon);
    }
    SECTION("RALS MODE = 3, Tucker + CP"){
      auto d = D3;
      CP_RALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff =
              A1.compress_compute_tucker(0.1, conv, 5, true, false, 100, true);
      CHECK((diff - results(14,0)) <= epsilon);
    }
    SECTION("RALS MODE = 3, Random + CP"){
      auto d = D3;
      CP_RALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff =
              A1.compress_compute_rand(2, conv, 0, 2, 5, true, false, 100, true);
      CHECK((diff - results(15,0)) <= epsilon);
    }
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
    SECTION("RALS MODE = 4, Tucker + CP"){
      auto d = D4;
      CP_RALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = A1.compress_compute_tucker(0.1, conv, 5, true, false, 100, true);
      CHECK((diff - results(18,0)) <= epsilon);
    }
    SECTION("RALS MODE = 4, Random + CP"){
      auto d = D4;
      CP_RALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = A1.compress_compute_rand(3, conv, 0, 2, 1, true, false, 20, true);
      CHECK((diff - results(19,0)) <= epsilon);
    }
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
    SECTION("RALS MODE = 5, Tucker + CP"){
      auto d = D5;
      CP_RALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = A1.compress_compute_tucker(0.1, conv, 5, true, false, 100, true);
      CHECK((diff - results(22,0)) <= epsilon);
    }
    SECTION("RALS MODE = 5, Random + CP"){
      auto d = D5;
      CP_RALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = A1.compress_compute_rand(1, conv, 0, 2, 5, true, false, 100, true);
      CHECK((diff - results(23,0)) <= epsilon);
    }
  }
  // CP-DF-ALS tests
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
  // coupled ALS test
  {
    SECTION("COUPLED-ALS MODE = 3, Finite rank"){
      conv_class_coupled conv_coupled(3, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D3, D3);
      conv_coupled.set_norm(norm3, norm3);
      double diff = A1.compute_rank(5, conv_coupled);
      CHECK((diff - results(30,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 3, Finite error"){
      conv_class_coupled conv_coupled(3, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D3, D3);
      conv_coupled.set_norm(norm3, norm3);
      double diff = A1.compute_error(conv_coupled, 1e-2, 1, 20);
      CHECK((diff - results(31,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 4, Finite rank"){
      conv_class_coupled conv_coupled(4, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D4, D4);
      conv_coupled.set_norm(norm4, norm4);
      double diff = A1.compute_rank(5,conv_coupled);
      CHECK((diff - results(32,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 4, Finite error"){
      conv_class_coupled conv_coupled(4, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D4, D4);
      conv_coupled.set_norm(norm4, norm4);
      double diff = A1.compute_error(conv_coupled, 1e-2, 1, 20);
      CHECK((diff - results(33,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 5, Finite rank"){
      conv_class_coupled conv_coupled(5, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D5, D5);
      conv_coupled.set_norm(norm5, norm5);
      double diff = A1.compute_rank(5,conv_coupled);
      CHECK((diff - results(34,0)) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 5, Finite error"){
      conv_class_coupled conv_coupled(5, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D5, D5);
      conv_coupled.set_norm(norm5, norm5);
      double diff = A1.compute_error(conv_coupled, 1e-2, 1, 20);
      CHECK((diff - results(35,0)) <= epsilon);
    }
  }
#endif
}
