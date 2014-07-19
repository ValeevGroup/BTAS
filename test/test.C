#include <iostream>
#include <algorithm>
#include <set>
#include <fstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;

#include <btas/range.h>
#include <btas/btas.h>
#include <btas/tensor.h>
#include <btas/tarray.h>
#include <btas/corange.h>
#include <btas/tensorview.h>
using namespace btas;

int main()
{
  //////////////////////////////////////////////////////////////////////////////
  // varray tests
  //////////////////////////////////////////////////////////////////////////////
  cout << "sizeof(btas::varray<double>) = " << sizeof(btas::varray<double>) << endl;
  cout << "sizeof(std::vector<double>) = " << sizeof(vector<double>) << endl;
  cout << "sizeof(std::tuple<double*,double*,std::allocator<double>>) = " << sizeof(tuple<double*,double*,std::allocator<double>>) << endl;
  cout << "sizeof(std::pair<double*,std::allocator<double>>) = " << sizeof(pair<double*,std::allocator<double>>) << endl;

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

    // test equality
    assert(r0 != r1);
    assert(r1 != r2);
    assert(congruent(r1,r2));
    Range1 r1copy(0, 5, 1);
    assert(r1 == r1copy);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Range tests
  //////////////////////////////////////////////////////////////////////////////

  const auto begin = std::array<long, 3>{{-1,-1,-1}};
  const auto size = std::array<std::size_t, 3>{{3,2,3}};
  // default (empty) Range
  Range x0;
  cout << "x0 = " << x0 << " area=" << x0.area() << endl;
  assert(x0.rank() == 0);  // std::cout << "rank = 0" << std::endl;

  // Range initialized by extents of each dimension
  Range x1(3, 2, 3);
  cout << "x1 = " << x1 << " area=" << x1.area() << endl;

  // Range initialized by extents of each dimension, given as an initializer list
  //Range x2 = {2, 3, 2};  // same as 'Range x2({2, 3, 2});'
  Range x2(4, 3, 2);
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
      cout << i << " ord=" << x2.ordinal(i) << endl;
    }
  }

  const auto p = array<size_t, 3>{{2,0,1}};
  auto x2p = permute(x2, p);
  cout << "x2 = " << x2 << endl;
  cout << "x2.permute(" << p << ") = " << x2p << endl;
  cout << "Iterating through permuted x2 using range-based for" << endl;
  for(auto i: x2p) {
    cout << i << " ord=" << x2p.ordinal(i) << endl;
  }

  cout << "Iterating through " << Range(2,3,4) << " using range-based for" << endl;
  for(auto i: Range(2,3,4)) {
    cout << i << endl;
  }

  Range x5 = {2, 3, 4};
  cout << "x5 = " << x5 << " area=" << x5.area() << endl;

  Range x6 ({-1, -1, -1}, {2, 3, 4});
  cout << "x6 = " << x6 << " area=" << x6.area() << endl;

  {
    typedef RangeNd<CblasRowMajor, std::array<size_t, 3>> Range3d;
    Range3d x;
    cout << "Static 3-d Range: x7 = " << x << " area=" << x.area() << endl;
    assert(x.rank() == 3);
    static_assert(x.rank() == 3, "default Range rank");
  }

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
    Tensor<std::array<complex<double>,3>> T5(2,3,4);
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
  axpy(0.5, S, U);// this segfaults because U is not initialized yet?

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
  Tensor<Tensor<double>> C(4,4); C.fill(Tensor<double>(2,2));// rank info is required to determine contraction ranks at gemm
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
    a.generate( [&v]() {v += 1.0; return v;});

    CMTensor b(3, 3);
    v = 0.0;
    b.generate( [&v]() {v += 1.0; return v;});

    CMTensor c(2, 3);
    gemm(CblasNoTrans, CblasTrans, 1.0, a, b, 0.0, c);
  }

  // test 11: serialization
  {
    const auto archive_fname = "test1.archive";

    Tensor<std::array<complex<double>,3>> T1(2,3,4);
    T1.fill({{{1.0,2.0}, {2.0,1.0}, {2.0,3.0} }});

    // write
    {
      std::ofstream os(archive_fname);
      assert(os.good());
      boost::archive::text_oarchive ar(os);
      ar << t; // fixed-size Tensor
      ar << A; // Tensor of Tensor
      ar << T1; // Tensor of complex datatypes
    }
    // read
    {
      std::ifstream is(archive_fname);
      assert(is.good());
      boost::archive::text_iarchive ar(is);

      TArray<double,3> tcopy;
      ar >> tcopy;

      Tensor<Tensor<double>> Acopy;
      ar >> Acopy; // Tensor of Tensor

      Tensor<std::array<complex<double>,3>> T1copy;
      ar >> T1copy; // Tensor of complex datatypes

      assert(t == tcopy);
      assert(A == Acopy);
      assert(T1 == T1copy);
    }
    std::remove(archive_fname);
  }

  //////////////////////////////////////////////////////////////////////////////
  // CoRange tests
  //////////////////////////////////////////////////////////////////////////////

  {
    Tensor<double> t0(2,3,4);
    Tensor<double> t1(1,3,4);
    auto r = make_corange(t0.range(), t1.range());
    cout << "co-iterating over CoRange of " << t0.range() << " and " << t1.range() << endl;
    for(auto i: r) {
      cout << first(i) << " " << second(i) << endl;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // TensorView tests
  //////////////////////////////////////////////////////////////////////////////

  {
    Tensor<double> t0(2,2,2);
    auto x=0.0; t0.generate([&]() { x += 1.0; return x;});

    { // vanilla iteration
      TensorView<double> t0v(t0);

      for(auto i: t0v)
        cout << i << endl;
    }

    { // test permuted view
      auto prange0 = permute(t0.range(),{2,1,0});
      // read only view
      auto t0v = make_cview(prange0, t0.storage());

      for(auto i: t0v)
        cout << i << endl;

      //
      // test const-correctness
      //

      //*(t0v.begin()) = -1.0; // error: assignment to read-only value

      // read-write view
      auto t0vw = make_view(prange0, t0.storage());
      *(t0vw.begin()) = -1.0; // OK: writable value

      // const TensorView behaves like TensorConstView
      const auto t0cv = make_view(prange0, t0.storage());
      //t0cv(0,0,0) = -1.0; // error: assignment to read-only value

      // const TensorView can be made from const Tensor
      const auto& t0_cref = t0;
      const auto t0cvr = make_view(prange0, t0_cref.storage());
      //t0cvr(0,0,0) = -1.0; // error: assignment to read-only value

      // making TensorView from const Tensor produces TensorConstView
      auto t0ncvr = make_view(prange0, t0_cref.storage());
      //t0ncvr(0,0,0) = -2.0; // error: assignment to read-only value N.B. Error message will be a bit more verbose here
    }

    //
    // more tests of const-correctness
    //
    {
      auto& t0_ref = t0;
      const auto& t0_cref = t0;
      TensorView<double> t0v1(t0_ref);
      TensorConstView<double> t0v2(t0_ref);
      //TensorView<double> t0v3(t0_cref); // compile error: nonconst view from const Tensor
      TensorConstView<double> t0v4(t0_cref);

      // make TensorConstView from TensorView
      TensorConstView<double> t0v5 = t0v1;

      //
      // test const-correctness of TensorRWView that tracks constness at (mostly) runtime
      //
      TensorRWView<double> t0v6 = t0v1;      // receive write access from TensorView
      t0v6(0,0,0) = -3.0;                    // OK
      TensorRWView<double> t0v6_copy = t0v6; // receive write access from writeable TensorRWView
      t0v6_copy(0,0,0) = -3.0;               // OK
      TensorRWView<double> t0v7 = t0v5;      // no write access from TensorConstView
      //t0v7(0,0,0) = -3.0;                  // runtime error: t0v7 has no write access
      const TensorRWView<double> t0v8 = t0v1;// receive write access from TensorView, but the object is const, hence can't write
      //t0v8(0,0,0) = -3.0;                  // compile error: t0v8 is const
      TensorRWView<double> t0v9 = t0v8;      // no write access: t0v8 has write access but is const!
      //t0v9(0,0,0) = -3.0;                  // runtime error: t0v9 has no write access

      TensorRWView<double> t0v10(t0_ref);    // write access when constructed directly from mutable Tensor
      t0v10(0,0,0) = -3.0;                   // OK
      TensorRWView<double> t0v11(t0_cref);   // no write access since Tensor is const
      //t0v11(0,0,0) = -3.0;                 // runtime error: t0v11 has no write access
    }

    {
      // read only view as a rank-2 tensor
      auto x = t0.extent();
      Range range_01_2(x[0] * x[1], x[2]);
      auto t0v = make_cview(range_01_2, t0.storage());

      for(auto i: make_corange(t0v.range(), t0v)) {
        cout << first(i) << " " << second(i) << endl;
      }

    }

    { // TensorView with different element type
      auto t0_float = make_cview<float>(t0.range(), t0.storage());
      auto t0_float1 = make_cview<float>(t0);
      auto t0_float2 = make_view<float>(t0);

      for(auto i: t0_float) {
        cout << i << endl;
      }

    }

  }

  //////////////////////////////////////////////////////////////////////////////
  // TensorMap tests
  //////////////////////////////////////////////////////////////////////////////

  {
      double* ptr0 = new double[24];
      for(auto i=0; i!=24; ++i) ptr0[i] = i;

      auto map0   = make_map(ptr0, Range{2, 3, 4});   // writable map
      auto map0c  = make_map(const_cast<const double*>(ptr0), Range{2, 3, 4});   // const map
      auto map0c2 = make_cmap(ptr0, Range{2, 3, 4}); // const map

      map0(1, 2, 3) = -1.0;
      //map0c(1, 2, 3) = -1.0;  // error: read-only map
      //map0c2(1, 2, 3) = -1.0; // error: read-only map

      std::cout << map0 << std::endl;
      for (const auto& i: map0) { std::cout << i << std::endl; }

      delete[] ptr0;
  }

  return 0;
}
