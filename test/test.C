#include <iostream>
using namespace std;

#include <tensor.h>
using namespace btas;

int main()
{
   Tensor<double> T(2,2,2);
   T.fill(0.0);
   T(1,1,1) = -0.5;
   return 0;
}
