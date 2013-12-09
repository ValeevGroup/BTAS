#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <functional>
#include <vector>
using namespace std;

#include <btas/tensor.h>
#include <btas/nditerator.h>
using namespace btas;

void print(const Tensor<double>& X)
{
   cout << "shape [";
   for(size_t i = 0; i < X.rank()-1; ++i)
   {
      cout << X.shape(i) << ",";
   }
   cout << X.shape(X.rank()-1) << "]" << endl;

   size_t str = X.shape(X.rank()-1);
   size_t ipr = 0;
   for(auto ix = X.begin(); ix != X.end(); ++ix, ++ipr)
   {
      if(ipr > 0 && ipr % str == 0) cout << endl;
      cout << setw(6) << fixed << *ix;
   }
   cout << endl;
}

// =================================================================================================
//
// Use cases of NDIterator
//
// =================================================================================================

int main()
{
   typedef Tensor<double>::shape_type shape_type;

   mt19937 rgen;
   uniform_real_distribution<double> dist(-1.0, 1.0);

// =================================================================================================
//
// create random tensor
//
// =================================================================================================

   Tensor<double> A(4,4,4); A.generate(bind(dist, rgen));

   cout.precision(2);
   cout << "Printing A: "; print(A);

// =================================================================================================
//
// permutation in terms of NDIterator
//
// =================================================================================================

   // [0,1,2] -> [2,1,0] i.e. [i,j,k] -> [k,j,i]
   shape_type permutation = { 2, 1, 0 };

   // permuted shape and stride hack
   shape_type new_shape (A.rank());
   shape_type new_stride(A.rank());
   for(size_t i = 0; i < A.rank(); ++i)
   {
      new_shape [i] = A.shape(permutation[i]);
      new_stride[i] = A.stride(permutation[i]);
   }

   // resize B with permuted shape
   Tensor<double> B(new_shape);

   // NDIterator ([shape], [stride], [start (iterator)], [current = start]);
   {
      NDIterator<Tensor<double>> ita(new_shape, new_stride, A.begin());
      for(auto itb = B.begin(); itb != B.end(); ++ita, ++itb)
      {
         *itb = *ita;
      }
   }

   cout.precision(2);
   cout << "Printing B: "; print(B);

// =================================================================================================
//
// slicing in terms of NDIterator
//
// =================================================================================================

   shape_type lower_bound = { 1, 1, 1 };
   shape_type slice_shape = { 2, 2, 2 };
   shape_type start_index = { 0, 0, 0 };

   // resize C with sliced shape
   Tensor<double> C(slice_shape);

   // NDIterator ([tensor], [start], [lower], [shape], [stride = tensor.stride()]);
   {
      NDIterator<Tensor<double>> ita(A, start_index, lower_bound, slice_shape);

      for(auto itc = C.begin(); itc != C.end(); ++ita, ++itc)
      {
         *itc = *ita;
      }
   }

   cout.precision(2);
   cout << "Printing C: "; print(C);

// =================================================================================================
//
// tie (or tracing) in terms of NDIterator
//
// =================================================================================================

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

// =================================================================================================
//
// slice and permute simultaneously
//
// =================================================================================================

   // permuted shape and stride hack
   shape_type cnew_shape (C.rank());
   shape_type cnew_stride(C.rank());
   for(size_t i = 0; i < C.rank(); ++i)
   {
      cnew_shape [i] = C.shape (permutation[i]);
      cnew_stride[i] = C.stride(permutation[i]);
   }

   // copy A into E
   Tensor<double> E(A);

   // iterator for slice of E
   {
      NDIterator<Tensor<double>> ite(E, start_index, lower_bound, slice_shape);
      NDIterator<Tensor<double>> ite_last = end(ite);

      // iterator for permute of slice of A
      NDIterator<Tensor<double>, NDIterator<Tensor<double>>> ita(cnew_shape, cnew_stride, NDIterator<Tensor<double>>(A, start_index, lower_bound, slice_shape));

      for(; ite != ite_last; ++ite, ++ita)
      {
         *ite = *ita;
      }
   }

   cout.precision(2);
   cout << "Printing E: "; print(E);

   NDIterator<Tensor<double>> it1(E.shape(), E.stride(), E.begin());

   NDConstIterator<Tensor<double>> it2(it1);

   return 0;
}
