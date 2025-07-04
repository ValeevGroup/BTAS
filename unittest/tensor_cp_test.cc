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
  using conv_class = btas::FitCheck<tensor>;
  using nocheck_conv = btas::NoCheck<tensor>;
  using appx_conv = btas::ApproxFitCheck<tensor>;
  using diff_conv = btas::DiffFitCheck<tensor>;
  using conv_class_coupled = btas::CoupledFitCheck<tensor>;
  using btas::CP_ALS;
  using btas::CP_RALS;
  using btas::CP_DF_ALS;
  using btas::COUPLED_CP_ALS;

  //double epsilon = fmax(1e-10, std::numeric_limits<double>::epsilon());
  double epsilon = 1e-5;
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

  tensor results(43, 1);
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

  conv_class conv(1e-7);
  nocheck_conv nocheck(1e-7);
  appx_conv appxcheck(1e-3);
  diff_conv diffcheck(1e-3);
  // ALS tests
  {
    SECTION("ALS MODE = 3, Finite rank"){
      CP_ALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff = A1.compute_rank(10, conv, 1, false, 0, 100, false, false, true);
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("ALS Mode = 3, Finite rank, no check"){
      CP_ALS<tensor, nocheck_conv> A1(D3);
      A1.compute_rank(10, nocheck, 1, false, 0, 100, false, false, true);
      auto apx = A1.reconstruct() - D3;
      auto diff =  sqrt(btas::dot(apx, apx)) / norm3;
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("ALS Mode = 3, Finite rank, approx check"){
      CP_ALS<tensor, appx_conv> A1(D3);
      A1.compute_rank(10, appxcheck, 1, false, 0, 100, false, false, true);
      auto apx = A1.reconstruct() - D3;
      auto diff =  sqrt(btas::dot(apx, apx)) / norm3;
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("ALS Mode = 3, Finite rank, diff fit check"){
      CP_ALS<tensor, diff_conv> A1(D3);
      A1.compute_rank(10, diffcheck, 1, false, 0, 100, false, false, true);
      auto apx = A1.reconstruct() - D3;
      auto diff =  sqrt(btas::dot(apx, apx)) / norm3;
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("ALS MODE = 3, Finite error"){
      CP_ALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff = A1.compute_error(conv, 1e-7, 1, 99);
      CHECK(std::abs(diff) <= epsilon);
    }
#if BTAS_ENABLE_TUCKER_CP_UT
    SECTION("ALS MODE = 3, Tucker + CP"){
      auto d = D3;
      btas::TUCKER_CP_ALS<tensor, conv_class> A1(d, 1e-3);
      conv.set_norm(norm3);
      double diff = A1.compute_rank(10, conv, 1, false,
                                    0, 100, false, false, true);
      CHECK(std::abs(diff) <= epsilon);
    }
#endif
#if BTAS_ENABLE_RANDOM_CP_UT
    SECTION("ALS MODE = 3, Random + CP"){
      auto d = D3;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff = 1.0 - A1.compress_compute_rand(2, conv, 0, 2, 5, true, false, 100, true);
      CHECK(std::abs(diff - results(3,0)) <= epsilon);
    }
#endif
    SECTION("ALS MODE = 4, Finite rank"){
      CP_ALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff = A1.compute_rank(55, conv, 1, true, 55);
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("ALS Mode = 4, Finite rank, no check"){
      CP_ALS<tensor, nocheck_conv> A1(D4);
      A1.compute_rank(55, nocheck, 1, true, 55, 50);
      auto apx = A1.reconstruct() - D4;
      auto diff =  sqrt(btas::dot(apx, apx)) / norm4;
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("ALS Mode = 3, Finite rank, approx check"){
      CP_ALS<tensor, appx_conv> A1(D4);
      A1.compute_rank(55, appxcheck, 1, true, 55, 50);
      auto apx = A1.reconstruct() - D4;
      auto diff =  sqrt(btas::dot(apx, apx)) / norm4;
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("ALS Mode = 3, Finite rank, diff fit check"){
      CP_ALS<tensor, diff_conv> A1(D4);
      A1.compute_rank(55, diffcheck, 1, true, 55, 50);
      auto apx = A1.reconstruct() - D4;
      auto diff =  sqrt(btas::dot(apx, apx)) / norm4;
      CHECK(std::abs(diff) <= epsilon);
    }

    SECTION("ALS MODE = 4, Finite error"){
      CP_ALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff = A1.compute_error(conv, 1e-2, 1, 57, true, 55);
      CHECK(std::abs(diff) <= epsilon);
    }
#if BTAS_ENABLE_TUCKER_CP_UT
    SECTION("ALS MODE = 4, Tucker + CP"){
      auto d = D4;
      btas::TUCKER_CP_ALS<tensor, conv_class> A1(d, 1e-3);
      conv.set_norm(norm4);
      double diff = A1.compute_rank(55, conv, 1, true, 55);
      CHECK(std::abs(diff) <= /* NB error too large with netlib blas on linux */ 3 * epsilon);
    }
#endif
#if BTAS_ENABLE_RANDOM_CP_UT
    SECTION("ALS MODE = 4, Random + CP"){
      auto d = D4;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = 1.0 - A1.compress_compute_rand(3, conv, 0, 2, 1, true, false, 20, true);
      CHECK(std::abs(diff - results(7,0)) <= epsilon);
    }
#endif
    SECTION("ALS MODE = 5, Finite rank"){
      CP_ALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_rank(57, conv, 1, true, 57);
      CHECK(std::abs(diff) <= epsilon);
    }
     SECTION("ALS MODE = 5, Finite error"){
      CP_ALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_error(conv, 1e-2, 1, 60, true, 57);
      CHECK(std::abs(diff) <= epsilon);
    }
#if BTAS_ENABLE_TUCKER_CP_UT
    SECTION("ALS MODE = 5, Tucker + CP"){
      auto d = D5;
      btas::TUCKER_CP_ALS<tensor, conv_class> A1(d, 1e-3);
      conv.set_norm(norm5);
      double diff = A1.compute_rank(67, conv, 1, true, 67);
      CHECK(std::abs(diff) <= epsilon);
    }
#endif
#if BTAS_ENABLE_RANDOM_CP_UT
    SECTION("ALS MODE = 5, Random + CP"){
      auto d = D5;
      CP_ALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = 1.0 - A1. compress_compute_rand(1, conv, 0, 2, 5, true, false, 100, false);
      CHECK(std::abs(diff - results(11,0)) <= epsilon);
    }
#endif
  }
  // RALS tests
  {
    SECTION("RALS MODE = 3, Finite rank") {
      CP_RALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff = A1.compute_rank(10, conv, 1, false, 0, 100, false, false, true);
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("RALS MODE = 3, Finite error") {
      CP_RALS<tensor, conv_class> A1(D3);
      conv.set_norm(norm3);
      double diff = A1.compute_error(conv, 1e-7, 1, 99);
      CHECK(std::abs(diff) <= epsilon);
    }
#if BTAS_ENABLE_TUCKER_CP_UT
    SECTION("RALS MODE = 3, Tucker + CP") {
      auto d = D3;

      btas::TUCKER_CP_RALS<tensor, conv_class> A1(d, 1e-3);
      conv.set_norm(norm3);
      double diff = A1.compute_rank(10, conv, 1, false, 0, 100, false, false, true);
      CHECK(std::abs(diff) <= epsilon);
    }
#endif
#if BTAS_ENABLE_RANDOM_CP_UT
    SECTION("RALS MODE = 3, Random + CP") {
      auto d = D3;
      CP_RALS<tensor, conv_class> A1(d);
      conv.set_norm(norm3);
      double diff = 1.0 - A1.compress_compute_rand(2, conv, 0, 2, 5, true, false, 100, true);
      CHECK(std::abs(diff - results(15, 0)) <= epsilon);
    }
#endif
    SECTION("RALS MODE = 4, Finite rank") {
      CP_RALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff = A1.compute_rank(55, conv, 1, true, 55);
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("RALS MODE = 4, Finite error") {
      CP_RALS<tensor, conv_class> A1(D4);
      conv.set_norm(norm4);
      double diff = A1.compute_error(conv, 1e-2, 1, 57, true, 55);
      CHECK(std::abs(diff) <= epsilon);
    }
#if BTAS_ENABLE_TUCKER_CP_UT
    SECTION("RALS MODE = 4, Tucker + CP") {
      auto d = D4;
      btas::TUCKER_CP_RALS<tensor, conv_class> A1(d, 1e-3);
      conv.set_norm(norm4);
      double diff = A1.compute_rank(55, conv, 1, true, 55);
      CHECK(std::abs(diff) <= /* NB error too large with netlib blas on linux */ 3 * epsilon);
    }
#endif
#if BTAS_ENABLE_RANDOM_CP_UT
    SECTION("RALS MODE = 4, Random + CP") {
      auto d = D4;
      CP_RALS<tensor, conv_class> A1(d);
      conv.set_norm(norm4);
      double diff = 1.0 - A1.compress_compute_rand(3, conv, 0, 2, 1, true, false, 20, true);
      CHECK(std::abs(diff - results(19, 0)) <= epsilon);
    }
#endif
    SECTION("RALS MODE = 5, Finite rank"){
      CP_RALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_rank(57, conv, 1, true, 57);
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("RALS MODE = 5, Finite error"){
      CP_RALS<tensor, conv_class> A1(D5);
      conv.set_norm(norm5);
      double diff = A1.compute_error(conv, 1e-2, 1, 60, true, 57);
      CHECK(std::abs(diff) <= epsilon);
    }
#if BTAS_ENABLE_TUCKER_CP_UT
    SECTION("RALS MODE = 5, Tucker + CP"){
      auto d = D5;
      btas::TUCKER_CP_RALS<tensor, conv_class > A1(d, 1e-1);
      conv.set_norm(norm5);
      double diff = A1.compute_rank(67, conv, 1, true, 67);
      CHECK(std::abs(diff) <= epsilon);
    }
#endif
#if BTAS_ENABLE_RANDOM_CP_UT
    SECTION("RALS MODE = 5, Random + CP"){
      auto d = D5;
      CP_RALS<tensor, conv_class> A1(d);
      conv.set_norm(norm5);
      double diff = 1.0 - A1.compress_compute_rand(1, conv, 0, 2, 5, true, false, 100, true);
      CHECK(std::abs(diff - results(23,0)) <= epsilon);
    }
#endif
  }

  // CP-DF-ALS tests
  // TODO I Think there is something wrong with CP-DF ALS solver with rank higher than 4.
  {
    SECTION("DF-ALS MODE = 3, Finite rank") {
      CP_DF_ALS<tensor, conv_class> A1(D3, D3);
      conv.set_norm(norm32);
      double diff = A1.compute_rank(40, conv, 1, true, 40);
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("DF-ALS MODE = 3, Finite error") {
      CP_DF_ALS<tensor, conv_class> A1(D3, D3);
      conv.set_norm(norm32);
      double diff = A1.compute_error(conv, 1e-8, 1, 43, true, 39);
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("DF-ALS MODE = 3, component decomposition") {
      CP_DF_ALS<tensor, conv_class> A1(D3, D3);
      conv.set_norm(norm32);
      double diff = A1.compute_comp_init(10, conv, 1000, true, false, true, 1e-2, 50);
      CHECK(std::abs(diff) <= epsilon);
    }
    /*SECTION("DF-ALS MODE = 4, Finite rank") {
      CP_DF_ALS<tensor, conv_class> A1(D4, D4);
      conv.set_norm(norm42);
      double diff = A1.compute_rank(500, conv, 1, true, 500);
      CHECK(std::abs(diff - 0.26010657273) <= epsilon);
    }
    SECTION("DF-ALS MODE = 4, Finite error") {
      CP_DF_ALS<tensor, conv_class> A1(D4, D4);
      conv.set_norm(norm42);
      double diff = A1.compute_error(conv, 1e-2, 1, 1000, true, 999);
      CHECK(std::abs(diff - 0.26010657273) <= epsilon);
    }
    SECTION("DF-ALS MODE = 4, component decompsoition") {
      CP_DF_ALS<tensor, conv_class> A1(D4, D4);
      conv.set_norm(norm42);
      double diff = A1.compute_comp_init(5, conv, 20);
      CHECK(std::abs(diff - results(29, 0)) <= epsilon);
    }*/
    /*SECTION("DF-ALS MODE = 5, Finite rank") {
      CP_DF_ALS<tensor, conv_class> A1(D5, D5);
      conv.set_norm(norm52);
      double diff = A1.compute_rank(900, conv, 1, true, 900);
      //CHECK(std::abs(diff - results(30, 0)) <= epsilon);
    }
    SECTION("DF-ALS MODE = 5, Finite error") {
      CP_DF_ALS<tensor, conv_class> A1(D5, D5);
      conv.set_norm(norm52);
      double diff = 1.0 - A1.compute_error(conv, 1e-2, 1, 1, false, 0, 100, false, true);
      CHECK(std::abs(diff - results(31, 0)) <= epsilon);
    }
    SECTION("DF-ALS MODE = 5, Component decomposition") {
      CP_DF_ALS<tensor, conv_class> A1(D5, D5);
      conv.set_norm(norm52);
      double diff = 1.0 - A1.compute_comp_init(5, conv, 20);
      CHECK(std::abs(diff - results(32, 0)) <= epsilon);
    }*/
  }
  // coupled ALS test
  {
    SECTION("COUPLED-ALS MODE = 3, Finite rank") {
      conv_class_coupled conv_coupled(3, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D3, D3);
      conv_coupled.set_norm(norm3, norm3);
      double diff = A1.compute_rank(10, conv_coupled);
      CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 3, Finite error") {
      conv_class_coupled conv_coupled(3, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D3, D3);
      conv_coupled.set_norm(norm3, norm3);
      double diff = A1.compute_error(conv_coupled, 1e-7, 1, 10);
      CHECK(std::abs(diff) <= epsilon);
    }
    // TODO These are harder problems and thus take too much effort for
    // unit tests. They decompose 2 order 4 and 2 order 5 tensors simultaneously
    // finding 7 and 9 unique factor matrices for each problem.
   /* SECTION("COUPLED-ALS MODE = 4, Finite rank") {
      conv_class_coupled conv_coupled(4, 1e-5);
      conv_coupled.verbose(true);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D4, D4);
      conv_coupled.set_norm(norm4, norm4);
      double diff = A1.compute_rank(320, conv_coupled, 1, true, 300);
      // CHECK(std::abs(diff) <= epsilon);
    }
    SECTION("COUPLED-ALS MODE = 4, Finite error") {
      conv_class_coupled conv_coupled(4, 1e-3);
      COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D4, D4);
      conv_coupled.set_norm(norm4, norm4);
      double diff = 1.0 - A1.compute_error(conv_coupled, 1e-2, 1, 19);
      CHECK(std::abs(diff - results(36, 0)) <= epsilon);
    }*/
    /*
   //TODO: Find more efficient ways to evaluate next two tests
   SECTION("COUPLED-ALS MODE = 5, Finite rank"){
     conv_class_coupled conv_coupled(5, 1e-3);
     COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D5, D5);
     conv_coupled.set_norm(norm5, norm5);
     double diff = 1.0 - A1.compute_rank(5,conv_coupled);
     CHECK(std::abs(diff - results(37,0)) <= epsilon);
   }
   SECTION("COUPLED-ALS MODE = 5, Finite error"){
     conv_class_coupled conv_coupled(5, 1e-3);
     COUPLED_CP_ALS<tensor, conv_class_coupled> A1(D5, D5);
     conv_coupled.set_norm(norm5, norm5);
     double diff = 1.0 - A1.compute_error(conv_coupled, 1e-2, 1, 19);
     CHECK(std::abs(diff - results(38,0)) <= epsilon);
   }
     */
  }
}
#endif
