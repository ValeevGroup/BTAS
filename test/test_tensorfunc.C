#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <functional>
#include <vector>
using namespace std;

#include <btas/tensor.h>
#include <btas/tensorview.h>
#include <btas/tensor_func.h>
using namespace btas;


template <typename T>
bool
isIn(T t, const std::vector<T>& v)
    {
    for(const auto& x : v) if(t == x) return true;
    return false;
    }

template<typename T>
T
vecMin(const std::vector<T>& v)
    {
    assert(!v.empty());
    auto res = v.front();
    for(auto x : v) res = std::min(res,x);
    return res;
    }


template <typename _Tensor>
void print(const _Tensor& X)
{
for(auto i : X.range()) 
    {
    cout << i << " " << X(i) << "\n";
    }
cout << endl;
}

template<typename _T>
const char*
typeToStr(const _T& T) { return "OtherType"; }

template<>
const char*
typeToStr(const Tensor<double>& T) { return "Tensor"; }

template<>
const char*
typeToStr(const TensorView<double>& T) { return "TensorView"; }

template<>
const char*
typeToStr(const TensorViewOf<const Tensor<double>>& T) { return "cTensorView"; }

// Set the elements of a Tensor T such that
// T(i,j,k) = 1ijk
// assuming individual dimensions are all less than 10.
// (The 1 in front is just a placeholder.)
void
fillEls(Tensor<double>& T)
{
  if(T.rank() == 0) return;
  const double base = pow(10.,T.rank());
  const size_t max_ii = T.rank()-1;
  for(auto I : T.range())
    {
    double &val = T(I);
    val = base;
    for(size_t ii = 0; ii <= max_ii; ++ii)
      {
      val += I[ii]*pow(10.,max_ii-ii);
      }
    }
}

int main()
{

cout << endl;

Tensor<double> A(3,3,3);
fillEls(A);
cout << typeToStr(A) << " A:" << endl;
print(A);

/////////////////////////////////////////
//
// permute(T,p)
//
/////////////////////////////////////////

cout << "\n----------------------------------------" << endl;
cout << "Testing permute" << endl;
cout << "----------------------------------------\n" << endl;

// [0,1,2] -> [2,1,0] i.e. [i,j,k] -> [k,j,i]
auto permutation = { 2, 1, 0 };

// create a permuted view of A; this creates a view only, no data copying
auto Av0 = permute(A, permutation);

cout << typeToStr(Av0) << " Av0 = permute(A," << permutation << "): " << endl; 
print(Av0);

Tensor<double> B0(Av0);
cout << typeToStr(B0) << " B0 = Av0: " << endl; 
print(B0);


cout << "Changing A(0,2,1) = -5" << endl;
A(0,2,1) = -5;
cout << "Changing Av0(0,2,2) = 7" << endl;
Av0(0,2,2) = 7;

TensorViewOf<const Tensor<double>> cAv0(Av0);
cout << typeToStr(cAv0) << " cAv0 (const view of A) = " << endl;
print(cAv0);

cout << "Changing A(0,2,1) = 1021" << endl;
cout << "Changing A(2,2,0) = 1220" << endl;
A(0,2,1) = 1021;
A(2,2,0) = 1220;

cout << "Beginning of A.storage() = " << &(A.storage()[0]) << endl;
cout << "Beginning of Av0.storage() = " << &(Av0.storage()[0]) << endl;
cout << "Beginning of cAv0.storage() = " << &(cAv0.storage()[0]) << endl;

print(A);

/////////////////////////////////////////
//
// diag(T)
//
/////////////////////////////////////////

cout << "\n----------------------------------------" << endl;
cout << "Testing diag" << endl;
cout << "----------------------------------------\n" << endl;

Range r({1,1,1},{3,3,4});
for(auto x : r) cout << x << " " << r.ordinal(x) << endl;
for(auto x : diag(r)) cout << x << " " << diag(r).ordinal(x) << endl; cout << endl;

RangeNd<CblasColMajor> rc({1,1,1},{3,3,4});
for(auto x : rc) cout << x << " " << rc.ordinal(x) << endl;
for(auto x : diag(rc)) cout << x << " " << diag(rc).ordinal(x) << endl; cout << endl;

auto d = diag(A);
cout << typeToStr(d) <<  " d = " << endl;
print(d);
cout << "Doing \"d(1) = 2\"" << endl;
d(1) = 2;
cout << "Now A(1,1,1) == " << A(1,1,1)  << endl;
cout << typeToStr(d) <<  " d = " << endl;
print(d);
cout << "Setting A(1,1,1) back to 1111" << "\n" << endl;
A(1,1,1) = 1111;

Tensor<double> dA = d;
cout << typeToStr(dA) <<  " dA = " << endl;
print(dA);

TensorViewOf<Tensor<double>> vA(A.range(),A.storage());
auto dvA = diag(vA);
cout << typeToStr(dvA) <<  " dvA = " << endl;
print(dvA);

//permute doesn't do anything when composed with diag, but just
//checking that it is possible
cout << typeToStr(diag(permute(A,{2,1,0}))) <<  " diag(permute(A,{2,1,0}) = " << endl;
print(diag(permute(A,{2,1,0})));

/////////////////////////////////////////
//
// group/flatten
//
/////////////////////////////////////////

cout << "\n----------------------------------------" << endl;
cout << "Testing group/flatten" << endl;
cout << "----------------------------------------\n" << endl;

//Range rr({0,0,0},{2,3,3});
//for(auto x : rr) cout << x << " " << rr.ordinal(x) << endl;
//
//auto gf = group(rr,0,2);
//cout << "gf = " << endl;
//for(auto x : gf) cout << x << " " << gf.ordinal(x) << endl;
//
//auto gb = group(rr,1,3);
//cout << "gb = " << endl;
//for(auto x : gb) cout << x << " " << gb.ordinal(x) << endl;
//
//auto ga = group(rr,0,3);
//cout << "ga = " << endl;
//for(auto x : ga) cout << x << " " << ga.ordinal(x) << endl;
//
//ga = flatten(rr);
//cout << "ga = " << endl;
//for(auto x : ga) cout << x << " " << ga.ordinal(x) << endl;
//
//Range f({0,0,0,0},{3,2,2,3});
//cout << "f = " << endl;
//for(auto x : f) cout << x << " " << f.ordinal(x) << endl;
//
//auto ff = group(f,0,2);
//cout << "ff = " << endl;
//for(auto x : ff) cout << x << " " << ff.ordinal(x) << endl;
//
//auto fm = group(f,1,3);
//cout << "fm = " << endl;
//for(auto x : fm) cout << x << " " << fm.ordinal(x) << endl;
//
//auto ff3 = group(f,0,3);
//cout << "ff3 = " << endl;
//for(auto x : ff3) cout << x << " " << ff3.ordinal(x) << endl;
//
//auto fe = group(f,2,4);
//cout << "fe = " << endl;
//for(auto x : fe) cout << x << " " << fe.ordinal(x) << endl;

Range rr1({1,1,1},{2,3,3});
for(auto x : rr1) cout << x << " " << rr1.ordinal(x) << endl;

auto gf1 = group(rr1,0,2);
cout << "gf1 = " << endl;
for(auto x : gf1) cout << x << " " << gf1.ordinal(x) << endl;

auto gb1 = group(rr1,1,3);
cout << "gb1 = " << endl;
for(auto x : gb1) cout << x << " " << gb1.ordinal(x) << endl;

auto ga1 = flatten(rr1);
cout << "ga1 = " << endl;
for(auto x : ga1) cout << x << " " << ga1.ordinal(x) << endl;

auto gAf = group(A,0,2);
cout << typeToStr(gAf) << " gAf = group(A,0,2): " << endl; 
print(gAf);

auto gAb = group(A,1,3);
cout << typeToStr(gAb) << " gAb = group(A,1,3): " << endl; 
print(gAb);

auto Af = flatten(A);
cout << typeToStr(Af) << " Af = flatten(A): " << endl; 
print(Af);

//
// 0,0,0  0
// 0,0,1  1
// 0,0,2  2
// 0,1,0  3
// 0,1,1  4
// 0,1,2  5
// 0,2,0  6
// 0,2,1  7
// 0,2,2  8
// 1,0,0  9
// 1,0,1  10
// 1,0,2  11
// 1,1,0  12
// 1,1,1  13
// 1,1,2  14
// 1,2,0  15
// 1,2,1  16
// 1,2,2  17
//
//  v v v
//
// 0,0  0
// 0,1  3
// 0,2  6
// 1,0  10
// 1,1  13
//
//
//
//
//
//

/////////////////////////////////////////
//
// tieIndex
//
/////////////////////////////////////////

cout << "\n----------------------------------------" << endl;
cout << "Testing tieIndex" << endl;
cout << "----------------------------------------\n" << endl;

Range rr({0,0,0,0},{3,4,4,4});
//RangeNd<CblasColMajor> rr({3,4,4,4});

std::vector<std::vector<size_t>> tests = {  {0},
                                            {1},
                                            {2},
                                            {3},
                                            {0,1},
                                            {0,2}, 
                                            {0,3}, 
                                            {1,2}, 
                                            {1,3}, 
                                            {2,3},
                                            {3,1},
                                            {3,2},
                                            {0,1,2}, 
                                            {0,1,3}, 
                                            {0,2,3}, 
                                            {1,2,3} 
                                          };
for(auto test : tests)
    {
    auto tr = tieIndex(rr,test);
    auto ti = vecMin(test);
    for(auto x : tr)
        {
        std::vector<size_t> y(rr.rank());
        size_t k = 0;
        for(size_t j = 0; j < rr.rank(); ++j)
            {
            if(isIn(j,test)) 
                { 
                if(j == ti) ++k;
                y[j] = x[ti]; 
                }
            else { y[j] = x[k]; ++k; }
            }
        if(tr.ordinal(x) != rr.ordinal(y)) 
            {
            cout << "Error for test = " << test << "; x = " << x << " " << tr.ordinal(x) << "; y = " << y << " " << rr.ordinal(y) << endl;
            }
        }

    }

//Range r2({0,0,0},{3,3,3});
//for(auto x : r2) cout << x << " " << r2.ordinal(x) << endl;
//auto tr = tieIndex(r2,0,1);
//for(auto x : tr) cout << x << " " << tr.ordinal(x) << endl;

auto tA1 = tieIndex(A,0,2);
print(tA1);

auto tA2 = tieIndex(A,0,1,2);
print(tA2);

auto tA3 = tieIndex(A,0,1);
print(tA3);

auto tA4 = tieIndex(A,1,2);
print(tA4);

std::array<size_t,2> inds = {1,2};
auto tA5 = tieIndex(A,inds);
print(tA5);

return 0;
}
