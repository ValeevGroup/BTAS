#include <iostream>
#include <algorithm>
using namespace std;

#include <tensor.h>
#include <varray.h>
using namespace btas;

int main()
{
   // test 1
   Tensor<double> T(2,2,2);
   fill(T.begin(), T.end(), 0.0);

   T(1,0,1) = -0.5;
   T.at(1,1,0) = 0.5;

   cout << "printing T: size = " << T.size() << " objsize = " << sizeof(T) << endl;
   for(double x : T) cout << x << endl;

   // test 2
   typedef Tensor<float, varray<float>, varray<int>> MyTensor;
   MyTensor::shape_type shape = { 4, 4 };
   MyTensor Q(shape);
   fill(Q.begin(), Q.end(), 1.0);

   MyTensor::shape_type index = { 1, 2 };

   Q(index) = -0.5;
   ++index[0];
   Q.at(index) = 0.5;

   cout << "printing Q: size = " << Q.size() << " objsize = " << sizeof(Q) << endl;
   for(double x : Q) cout << x << endl;

// // test 3
// Slice<MyTensor>::index_type lb = { 1, 1 };
// Slice<MyTensor>::index_type ub = { 3, 3 };
// Slice<MyTensor> S(Q, lb, ub);

// cout << "printing S: size = " << S.size() << endl;
// for(double x : S) cout << x << endl;

   return 0;
}
