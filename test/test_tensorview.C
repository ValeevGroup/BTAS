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

template <typename _Tensor>
void print(const _Tensor& X)
{
  typedef typename _Tensor::range_type range_type;
  typedef typename _Tensor::value_type value_type;
  constexpr auto order = boxrange_iteration_order<range_type>::value;

   cout << "range = " << X.range() << endl;

   auto ldx = X.extent(order == boxrange_iteration_order<range_type>::column_major ? 0 : X.rank()-1);

   auto i = size_t{1};
   for(const auto& e: X)
   {
      cout << setw(6) << fixed << e;
      if(i % ldx == 0) cout << endl;
      else cout << " ";
      ++i;
   }
   cout << endl;
}

// =================================================================================================
//
// Use cases of Tensor
//
// =================================================================================================

int main()
{
   mt19937 rgen;
   auto dist = uniform_real_distribution<double>{-1.0, 1.0};

// =================================================================================================
//
// create random tensor
//
// =================================================================================================

   const auto n = 3;
   Tensor<double> A(n,n,n); A.generate(bind(dist, rgen));

   cout.precision(2);
   cout << "Tensor A: "; print(A);

// =================================================================================================
//
// permutation
//
// =================================================================================================

   // [0,1,2] -> [2,1,0] i.e. [i,j,k] -> [k,j,i]
   auto permutation = { 2, 1, 0 };

   // create a permuted view of A; this creates a view only, no data copying
   auto Av0 = permute(A, permutation);

   cout.precision(2);
   cout << "TensorView Av0 = permute(" << permutation << ",A): "; print(Av0);

   // create a permuted Tensor from permuted view of A
   Tensor<double> B(Av0);
   cout << "Tensor B = permute(" << permutation << ",A): "; print(B);

   *(++Av0.begin()) = std::numeric_limits<double>::quiet_NaN();
   cout << "TensorView Av0 after fiddling: "; print(Av0);
   cout << "Tensor A after fiddling with Av0: "; print(A);
   cout << "Tensor B after fiddling with Av0: "; print(B);

   // create a permuted Tensor from A directly
   Tensor<double> C = permute(A, {2,0,1});
   cout << "Tensor C = permute(" << std::array<int,3>({2,0,1}) << ",A): "; print(C);

// =================================================================================================
//
// slicing
//
// =================================================================================================
   auto slice_lobound = { 1, 1, 1 };
   auto slice_upbound = { n, n, n };

   TensorView<double> Dv0( A.range().slice(slice_lobound, slice_upbound) , A.storage());
   auto Dv1 = permute(A, {2,0,1});

   cout << "Printing Dv0 = slice(A," << slice_lobound << "," << slice_upbound << "): "; print(Dv0);
   cout << "Printing Dv1 = permute(A,{2,0,1}): "; print(Dv1);

   // slice and permute
   TensorView<double> Dv2( permute(A.range().slice(slice_lobound, slice_upbound), {2,0,1}) , A.storage());

   cout << "Printing Dv2 = permute(slice(A," << slice_lobound << "," << slice_upbound << "),"
       << std::array<int,3>({2,0,1}) << ": "; print(Dv2);

   // permute and slice
   TensorView<double> Dv3( permute(A.range(), {2,0,1}).slice(slice_lobound, slice_upbound) , A.storage());

   cout << "Printing Dv3 = slice(permute(A," << std::array<int,3>({2,0,1}) << "),"<< slice_lobound << "," << slice_upbound << ")"
        << ": "; print(Dv3);

   // Dv2 should be equal Dv3 because it's a cubic range, break the symmetry to see the difference

// =================================================================================================
//
// tie (or tracing) in terms of NDIterator
//
// =================================================================================================
#if 0
   // A(i,j,i) -> D(i,j)
   size_t idim = A.shape(0);
   size_t jdim = A.shape(1);
   size_t istr = A.stride(0)+A.stride(2);
   size_t jstr = A.stride(1);
   shape_type tie_shape  = { idim, jdim };
   shape_type tie_stride = { istr, jstr };

   // resize D with tied shape
   Tensor<double> D(tie_shape);

   // NDIterator ([shape], [stride], [start (pointer)], [current = start]);
   {
      NDIterator<Tensor<double>, const double*> ita(tie_shape, tie_stride, A.data());

      for(auto itd = D.begin(); itd != D.end(); ++itd, ++ita)
      {
         *itd = *ita;
      }
   }

   cout.precision(2);
   cout << "Printing D: "; print(D);
#endif

   return 0;
}
