#ifdef BTAS_HAS_CBLAS
#include "test.h"

#include <iostream>
#include <fstream>
#include "btas/btas.h"

using namespace btas;
typedef btas::Tensor<double> tensor;
double epsilon = fmax(1e10, std::numeric_limits<double>::epsilon());
TEST_CASE("Dimension 3")
{
  tensor D3(5,2,9);
  std::ifstream in;
  in.open("./mat3D.txt");
  if(!in.is_open()){
    std::cout << "File isn't open" << std::endl;
    return;
  }
  for(auto& i: D3){
    in >> i;
  }
  in.close();
  tensor results(12,1);
  std::ifstream res("./mat3D_res.txt", std::ifstream::in);
  if(!res.is_open()){
    std::cout << "Results are not open " << std::endl;
    return;
  }
  for(auto& i: results){
    res >> i;
  }
  SECTION("Finite rank")
  {
    {
      CP_ALS<tensor> A1(D3);
      double diff = A1.compute(5,false,true, 1e3, 1, 1e-4);
      CHECK((diff - results(0,0)) <= epsilon);
    }
  }
	SECTION("Finite error")
	{
		{
      CP_ALS<tensor> A1(D3);
      double diff = A1.compute(1e-2, true, 1e3, 1, 1e-4);
      CHECK((diff - results(4,0)) <= epsilon);
    }
	}
}

TEST_CASE("Dimension 4")
{
  tensor D4(6,3,3,7);
  std::ifstream in;
  in.open("./mat4D.txt");
  if(!in.is_open()){
    std::cout << "File isn't open" << std::endl;
    return;
  }
  for(auto& i: D4){
    in >> i;
  }
  in.close();
  tensor results(12,1);
  std::ifstream res("./mat4D_res.txt", std::ifstream::in);
  if(!res.is_open()){
    std::cout << "Results are not open " << std::endl;
    return;
  }
  for(auto& i: results){
    res >> i;
  }
	SECTION("Finite rank")
	{
    {
      CP_ALS<tensor> A1(D4);
      double diff = A1.compute(5,false, true, 1e3, 1, 1e-4);
      CHECK((diff - results(0,0)) <= epsilon);
    }
	}
	SECTION("Finite error")
	{
	  {
      CP_ALS<tensor> A1(D4);
      double diff = A1.compute(1e-2, true, 1e3, 1, 1e-4);
      CHECK((diff - results(4,0)) <= epsilon);
    }
  }
}

TEST_CASE("Dimension 5")
{
  tensor D5(2,6,1,9,3);
  std::ifstream in;
  in.open("./mat5D.txt");
  if(!in.is_open()){
    std::cout << "File isn't open" << std::endl;
    return;
  }
  for(auto& i: D5){
    in >> i;
  }
  in.close();
  tensor results(12,1);
  std::ifstream res("./mat5D_res.txt", std::ifstream::in);
  if(!res.is_open()){
    std::cout << "Results are not open " << std::endl;
    return;
  }
  for(auto& i: results){
    res >> i;
  }
  SECTION("Finite rank")
  {
    {
      CP_ALS<tensor> A1(D5);
      double diff = A1.compute(5,false, false, 1e3, 1, 1e-4);
      CHECK((diff - results(0,0)) <= epsilon);
    }
  }
  SECTION("Finite error")
  {
    {
      CP_ALS<tensor> A1(D5);
      double diff = A1.compute(1e-2, true, 1e3, 1, 1e-4);
      CHECK((diff - results(4,0)) <= epsilon);
    }
  }
}
#endif //btas_has_cblas
