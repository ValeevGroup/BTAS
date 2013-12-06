#include <iostream>
#include <algorithm>
#include <set>
using namespace std;

//namespace btas { enum CBLAS_TRANSPOSE { CblasNoTrans, CblasTrans, CblasConjTrans }; };

#include <btas/varray.h>
#include <btas/btas.h>
#include <btas/tensor.h>
#include <btas/tarray.h>
using namespace btas;

int main()
{
  // test 0
  {
    Tensor<double> T0;
    Tensor<int> T1(2);
    Tensor<bool> T2(2, 2);
    Tensor<std::complex<int> > T3(2, 2, 2);
    Tensor<std::complex<double> > T4(2, 2, 2, 2);
  }

   // test 1
   Tensor<double> T(2,2,2); T.fill(0.0);

   T(1,0,1) = -0.5;

   T.at(1,1,0) = 0.5;

   cout << "printing T: size = " << T.size() << " objsize = " << sizeof(T) << endl;
   for(double x : T) cout << x << endl;

   // test 2
   typedef Tensor<float, btas::DEFAULT_RANGE, varray<float>> MyTensor;
   MyTensor::range_type range(4, 4);
   MyTensor Q(range); Q.fill(2.0);

   MyTensor::index_type index = {1, 2};

   Q(index) = -0.5;
   ++index[0];
   Q.at(index) = 0.5;

   cout << "printing Q: size = " << Q.size() << " objsize = " << sizeof(Q) << endl;
   for(double x : Q) cout << x << endl;

   Q = T;
   cout << "printing Q (=T): size = " << Q.size() << " objsize = " << sizeof(Q) << endl;
   for(double x : Q) cout << x << endl;

   // test 3
   Tensor<double> S(2,2,2); S.fill(1.0);
   axpy(0.5, T, S);

   cout << "printing S: size = " << S.size() << " objsize = " << sizeof(S) << endl;
   for(double x : S) cout << x << endl;

   Tensor<double> U;
   axpy(0.5, S, U); // this segfaults because U is not initialized yet?

   cout << "printing U: size = " << U.size() << " objsize = " << sizeof(U) << endl;
   for(double x : U) cout << x << endl;

#if 1
   // test 4
   Tensor<double> V(0,0);
   gemm(CblasNoTrans, CblasNoTrans, 1.0, T, S, 1.0, V);

   cout << "printing V: size = " << V.size() << " objsize = " << sizeof(V) << endl;
   for(double x : V) cout << x << endl;
#endif

   // test 5
   cout << boolalpha;
   cout << "is_tensor<Tensor<double>> = " << is_tensor<Tensor<double>>::value << endl;
   cout << "is_tensor<Tensor<Tensor<double>>> = " << is_tensor<Tensor<Tensor<double>>>::value << endl;
   cout << "is_tensor<vector<double>> = " << is_tensor<vector<double>>::value << endl;

   // test 6
   Tensor<Tensor<double>> A(4,4);
   Tensor<double> aval(2,2); aval.fill(1.0); A.fill(aval);
   Tensor<Tensor<double>> B(4,4);
   Tensor<double> bval(2,2); bval.fill(2.0); B.fill(bval);
   Tensor<Tensor<double>> C(4,4); C.fill(Tensor<double>(2,2)); // rank info is required to determine contraction ranks at gemm
   gemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 1.0, C);

   // test 7
   Tensor<double> a(4,4); a.fill(1.0);
// gemm(CblasNoTrans, CblasNoTrans, 1.0, A, a, 1.0, C); // this will give a compile-time error, since gemm for "tensor of tensor" and "tensor" is not supported

   // test 8
   TArray<double,3> t(2,2,2); t.fill(0.0);

   t(1,0,1) = -0.5;

   t.at(1,1,0) = 0.5;

   cout << "printing t: size = " << t.size() << " objsize = " << sizeof(t) << endl;
   for(double x : t) cout << x << endl;

#if 0 // TArray not implemented
   TArray<double,3> s(S);

   cout << "printing s: size = " << s.size() << " objsize = " << sizeof(s) << endl;
   for(double x : s) cout << x << endl;

   TArray<double,2> v;
   gemm(CblasNoTrans, CblasNoTrans, 1.0, t, s, 1.0, v);

   cout << "printing v: size = " << v.size() << " objsize = " << sizeof(v) << endl;
   for(double x : v) cout << x << endl;

   TArray<double,3,std::set<double>> u;
#endif

   return 0;
}
