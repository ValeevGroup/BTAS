#include <iostream>
#include <algorithm>
using namespace std;

#include <tensor.h>
#include <varray.h>
#include <axpy_impl.h>
using namespace btas;

int main()
{
   // test 1
   Tensor<double> T(2,2,2); T.fill(0.0);

   T(1,0,1) = -0.5;

   T.at(1,1,0) = 0.5;

   cout << "printing T: size = " << T.size() << " objsize = " << sizeof(T) << endl;
   for(double x : T) cout << x << endl;

   // test 2
   typedef Tensor<float, varray<float>, varray<int>> MyTensor;
   MyTensor::shape_type shape = { 4, 4 };
   MyTensor Q(shape); Q.fill(2.0);

   MyTensor::shape_type index = { 1, 2 };

   Q(index) = -0.5;
   ++index[0];
   Q.at(index) = 0.5;

   cout << "printing Q: size = " << Q.size() << " objsize = " << sizeof(Q) << endl;
   for(double x : Q) cout << x << endl;

   // test 3
   Tensor<double> S(2,2,2); S.fill(1.0);
   axpy(0.5, T, S);

   cout << "printing S: size = " << S.size() << " objsize = " << sizeof(S) << endl;
   for(double x : S) cout << x << endl;

   Tensor<double> U;
   axpy(0.5, S, U);

   cout << "printing U: size = " << U.size() << " objsize = " << sizeof(U) << endl;
   for(double x : U) cout << x << endl;

   return 0;
}
