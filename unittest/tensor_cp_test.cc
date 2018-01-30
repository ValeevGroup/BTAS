#ifdef BTAS_HAS_CBLAS
#include "test.h"

#include <iostream>
#include <fstream>
#include "btas/btas.h"

using namespace btas;
typedef btas::Tensor<double> tensor;
double epsilon = fmax(1e-10, std::numeric_limits<double>::epsilon());
TEST_CASE("CP_ALS"){
	tensor D3(5,2,9);
  std::ifstream in3;
  in3.open("./mat3D.txt");
  if(!in3.is_open()){
    std::cout << "File isn't open" << std::endl;
     return;
  }
  for(auto& i: D3){
    in3 >> i;
  }
  in3.close();

  tensor D4(6,3,3,7);
  std::ifstream in4;
  in4.open("./mat4D.txt");
  if(!in4.is_open()){
    std::cout << "File isn't open" << std::endl;
     return;
  }
  for(auto& i: D4){
    in4 >> i;
  }
  in4.close();

  tensor D5(2,6,1,9,3);
  std::ifstream in5;
  in5.open("./mat5D.txt");
  if(!in5.is_open()){
    std::cout << "File isn't open" << std::endl;
     return;
  }
  for(auto& i: D5){
    in5 >> i;
  }
  in5.close();

  tensor results(12,1);
  std::ifstream res("./cp_test_results.txt", std::ifstream::in);
  if(!res.is_open()){
    std::cout << "Results are not open " << std::endl;
    return;
  }
  for(auto& i: results){
    res >> i;
  }

  SECTION("MODE = 3, Finite rank"){
  	{
      CP_ALS<tensor> A1(D3);
      double diff = A1.compute_rank(5,false,true, 1, 1e3, 1e-2);
      CHECK((diff - results(0,0)) <= epsilon);
    }
  }
  SECTION("MODE = 3, Finite error"){
  	{
      CP_ALS<tensor> A1(D3);
      double diff = A1.compute_error(1e-2, true, 1, 1e3, 1e-2);
      CHECK((diff - results(1,0)) <= epsilon);
    }
  }
#ifdef _HAS_INTEL_MKL
  SECTION("MODE = 3, Tucker + CP"){
  	{
      CP_ALS<tensor> A1(D3);
      double diff = A1.compress_compute_tucker(.1, false, 1e-2, 5, true, true);
      CHECK((diff - results(2,0)) <= epsilon);
    }
  }
  SECTION("MODE = 3, Random + CP"){
  	{
      CP_ALS<tensor> A1(D3);
      double diff = A1.compress_compute_rand(2, 0, 2, false, 1e-2, 5, true, true );
      CHECK((diff - results(3,0)) <= epsilon);
    }
  }
#endif
	
  SECTION("MODE = 4, Finite rank"){
  	{
      CP_ALS<tensor> A1(D4);
      double diff = A1.compute_rank(5,false,true, 1, 1e3, 1e-2);
      CHECK((diff - results(4,0)) <= epsilon);
    }
  }
  SECTION("MODE = 4, Finite error"){
  	{
      CP_ALS<tensor> A1(D4);
      double diff = A1.compute_error(1e-2, true, 1, 1e3, 1e-2);
      CHECK((diff - results(5,0)) <= epsilon);
    }
  }
#ifdef _HAS_INTEL_MKL
  SECTION("MODE = 4, Tucker + CP"){
  	{
      CP_ALS<tensor> A1(D4);
      double diff = A1.compress_compute_tucker(.1, false, 1e-2, 5, true, true);
      CHECK((diff - results(6,0)) <= epsilon);
    }
  }
  SECTION("MODE = 4, Random + CP"){
  	{
      CP_ALS<tensor> A1(D4);
      double diff = A1.compress_compute_rand(3, 0, 2, true, 1e-2 );      
      CHECK((diff - results(7,0)) <= epsilon);
    }
  }
#endif
	
  SECTION("MODE = 5, Finite rank"){
  	{
      CP_ALS<tensor> A1(D5);
      double diff = A1.compute_rank(5,false,true, 1, 1e3, 1e-2);
      CHECK((diff - results(8,0)) <= epsilon);
    }
  }
  SECTION("MODE = 5, Finite error"){
  	{
      CP_ALS<tensor> A1(D5);
      double diff = A1.compute_error(1e-2, true, 1, 1e3, 1e-2);
      CHECK((diff - results(9,0)) <= epsilon);
    }
  }
#ifdef _HAS_INTEL_MKL
  SECTION("MODE = 5, Tucker + CP"){
  	{
      CP_ALS<tensor> A1(D5);
      double diff = A1.compress_compute_tucker(.1, false, 1e-2, 5, true, true);
      CHECK((diff - results(10,0)) <= epsilon);
    }
  }
  SECTION("MODE = 5, Random + CP"){
  	{
      CP_ALS<tensor> A1(D5);
      double diff = A1. compress_compute_rand(1,0);
      CHECK((diff - results(11,0)) <= epsilon);
    }
  }
#endif
}
#endif //BTAS_HAS_CBLAS
