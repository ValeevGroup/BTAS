#include <iostream>
#include <algorithm>
#include <set>
using namespace std;

#include <btas/range.h>
#include <btas/btas.h>
#include <btas/tensor.h>
#include <btas/tarray.h>
using namespace btas;

int main()
{
  //////////////////////////////////////////////////////////////////////////////
  // Range1 tests
  //////////////////////////////////////////////////////////////////////////////
  {
    Range1 r0;         // empty = {}
    cout << "r0 = " << r0 << endl;
    Range1 r1(5);      // [0,5) = {0, 1, 2, 3, 4}
    cout << "r1 = " << r1 << endl;
    Range1 r2(-1,4);    // [-1,4) = {-1, 0, 1, 2, 3}
    cout << "r2 = " << r2 << endl;
    Range1 r3(-1,7,2);  // [-1,7) with stride 2 = {-1, 1, 3, 5}
    cout << "r3 = " << r3 << endl;
    {
      cout << "Iterating through r3 using range-based for" << endl;
      for(auto i: r3) {
        cout << i << endl;
      }
    }

    Range1 r4 = {};           r4 = {};
    Range1 r5 = {3};          r5 = {2};
    Range1 r6 = {-3, 3};      r6 = {2, 7};
    Range1 r7 = {-3, 13, 2};  r7 = {-1, 10, 2};

  }

  //////////////////////////////////////////////////////////////////////////////
  // Range tests
  //////////////////////////////////////////////////////////////////////////////

  std::array<long, 3> begin = {-1,-1,-1};
  std::array<std::size_t, 3> size = {3,2,3};
  // default (empty) Range
  Range x0;
  cout << "x0 = " << x0 << " area=" << x0.area() << endl;

  // Range initialized by extents of each dimension
  Range x1(3, 2, 3);
  cout << "x1 = " << x1 << " area=" << x1.area() << endl;

  // Range initialized by extents of each dimension, given as an initializer list
  //Range x2 = {2, 3, 2};  // same as 'Range x2({2, 3, 2});'
  Range x2(2, 3, 2);
  cout << "x2 = " << x2 << " area=" << x1.area() << endl;

  // fixed-rank Range
  RangeNd<CblasRowMajor, array<long, 3> > x3(begin, size);
  cout << "x3 = " << x3 << " area=" << x3.area() << endl;

  // col-major std::vector-based Range
  RangeNd<CblasColMajor, vector<long> > x4(size);
  cout << "x4 = " << x4 << " area=" << x4.area() << endl;

  {
    cout << "Iterating through x1 using iterator-based for" << endl;
    for(auto i=x1.begin(); i!=x1.end(); ++i) {
      cout << *i << endl;
    }
  }

  {
    cout << "Iterating through x2 using range-based for" << endl;
    for(auto i: x2) {
      cout << i << endl;
    }
  }

  cout << "Iterating through " << Range(2,3,4) << " using range-based for" << endl;
  for(auto i: Range(2,3,4)) {
    cout << i << endl;
  }

  Range x5 = {2, 3, 4};
  cout << "x5 = " << x5 << " area=" << x5.area() << endl;

  Range x6 ({-1, -1, -1}, {2, 3, 4});
  cout << "x6 = " << x6 << " area=" << x6.area() << endl;

  //////////////////////////////////////////////////////////////////////////////
  // Tensor tests
  //////////////////////////////////////////////////////////////////////////////

  // test 0: constructors
  {
    Tensor<double> T0;
    Tensor<int> T1(2);
    Tensor<bool> T2(2, 2);
    Tensor<std::complex<int> > T3(2, 2, 2);
    Tensor<std::complex<double> > T4(2, 2, 2, 2);
  }

   // test 1: random access
   Tensor<double> T(2,2,2); T.fill(0.0);

   T(1,0,1) = -0.5;

   T.at(1,1,0) = 0.5;

   cout << "printing T: size = " << T.size() << " objsize = " << sizeof(T) << endl;
   for(double x : T) cout << x << endl;

   // test 2: iteration
   typedef Tensor<float, Range, varray<float>> MyTensor;
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

   // test 3: axpy
   Tensor<double> S(2,2,2); S.fill(1.0);
   axpy(0.5, T, S);

   cout << "printing S: size = " << S.size() << " objsize = " << sizeof(S) << endl;
   for(double x : S) cout << x << endl;

   Tensor<double> U;
   axpy(0.5, S, U); // this segfaults because U is not initialized yet?

   cout << "printing U: size = " << U.size() << " objsize = " << sizeof(U) << endl;
   for(double x : U) cout << x << endl;

   // test 4: gemm
   Tensor<double> V(0,0);
   gemm(CblasNoTrans, CblasNoTrans, 1.0, T, S, 1.0, V);

   cout << "printing V: size = " << V.size() << " objsize = " << sizeof(V) << endl;
   for(double x : V) cout << x << endl;
   // test 5: tensor of tensor (ToT)
   cout << boolalpha;
   cout << "is_tensor<Tensor<double>> = " << is_tensor<Tensor<double>>::value << endl;
   cout << "is_tensor<Tensor<Tensor<double>>> = " << is_tensor<Tensor<Tensor<double>>>::value << endl;
   cout << "is_tensor<vector<double>> = " << is_tensor<vector<double>>::value << endl;

   // test 6: ToT operations
   Tensor<Tensor<double>> A(4,4);
   Tensor<double> aval(2,2); aval.fill(1.0); A.fill(aval);
   Tensor<Tensor<double>> B(4,4);
   Tensor<double> bval(2,2); bval.fill(2.0); B.fill(bval);
   Tensor<Tensor<double>> C(4,4); C.fill(Tensor<double>(2,2)); // rank info is required to determine contraction ranks at gemm
   gemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 1.0, C);

   // test 7: argument checking in gemm
   Tensor<double> a(4,4); a.fill(1.0);
// gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1.0, A, a, 1.0, C); // this will give a compile-time error, since gemm for "tensor of tensor" and "tensor" is not supported

   // test 8: fixed-rank tensor
   TArray<double,3> t(2,2,2); t.fill(0.0);

   t(1,0,1) = -0.5;

   t.at(1,1,0) = 0.5;

   cout << "printing t: size = " << t.size() << " objsize = " << sizeof(t) << endl;
   for(double x : t) cout << x << endl;

   TArray<double,3> s(S);

   cout << "printing s: size = " << s.size() << " objsize = " << sizeof(s) << endl;
   for(double x : s) cout << x << endl;

   TArray<double,2> v;
   gemm(CblasNoTrans, CblasNoTrans, 1.0, t, s, 1.0, v);

   cout << "printing v: size = " << v.size() << " objsize = " << sizeof(v) << endl;
   for(double x : v) cout << x << endl;

   TArray<double,3,CblasRowMajor,std::set<double>> u;

   cout << "dot(a, a) = " << dot(a, a) << endl;

   // test 9: fixed-size tensor
   {
     // to avoid dynamic allocation and memory overheads, use std::array
     typedef Tensor<double, RangeNd<CblasRowMajor, std::array<char, 2> >, std::array<double, 9> > MyTensor;
     MyTensor::range_type range(3, 3);
     //MyTensor::range_type range(4, 4); // runtime-error with this range -- bigger than storage
     MyTensor Q(range); Q.fill(2.0);
   }

   // test 10: gemm with col-major tensors
   {
     const CBLAS_ORDER order = CblasColMajor;
     typedef RangeNd<order> CMRange;
     typedef Tensor<double, CMRange > CMTensor;

     CMTensor a(2, 3);
     auto v = 0.0;
     a.generate( [&v]() { v += 1.0; return v; } );

     CMTensor b(3, 3);
     v = 0.0;
     b.generate( [&v]() { v += 1.0; return v; } );

     CMTensor c(2, 3);
     gemm(CblasNoTrans, CblasTrans, 1.0, a, b, 0.0, c);
   }

   return 0;
}
