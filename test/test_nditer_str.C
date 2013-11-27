#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <functional>
#include <vector>
#include <string>
#include <sstream>
using namespace std;

#include <btas/tensor.h>
#include <btas/nditerator.h>
using namespace btas;

void print(const Tensor<string>& X)
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
      cout << " " << *ix;
   }
   cout << endl;
}

// =================================================================================================
//
// This is for checking how NDIterator works by monitoring explicit indexing
//
// =================================================================================================

int main()
{
   typedef Tensor<string>::shape_type shape_type;

// mt19937 rgen;
// uniform_real_distribution<string> dist(-1.0, 1.0);

// =================================================================================================
//
// create random tensor
//
// =================================================================================================

   Tensor<string> A(4,4,4); // A.generate(bind(dist, rgen));
   for(size_t i = 0; i < A.shape(0); ++i) {
      for(size_t j = 0; j < A.shape(1); ++j) {
         for(size_t k = 0; k < A.shape(2); ++k) {
            ostringstream sout;
            sout << "[" << i << "," << j << "," << k << "]";
            A(i,j,k) = sout.str();
         }
      }
   }

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
      new_shape[i] = A.shape(permutation[i]);
      new_stride[i] = A.stride(permutation[i]);
   }

   // resize B with permuted shape
   Tensor<string> B(new_shape);

   // NDIterator ([pointer to start], [new shape], [stride hack], [current index]);
   NDIterator<string*, shape_type> it_permt(A.data(), new_shape, new_stride);

   for(auto ib = B.begin(); ib != B.end(); ++ib, ++it_permt)
   {
      *ib = *it_permt;
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

   // calculate offset of starting address
   size_t offset = 0;
   for(size_t i = 0; i < A.rank(); ++i)
   {
      offset += lower_bound[i]*A.stride(i);
   }

   // resize C with sliced shape
   Tensor<string> C(slice_shape);

   // NDIterator ([pointer to start], [new shape], [stride hack], [current index]);
   NDIterator<string*, shape_type> it_slice(A.data()+offset, slice_shape, A.stride());

   for(auto ic = C.begin(); ic != C.end(); ++ic, ++it_slice)
   {
      *ic = *it_slice;
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
   Tensor<string> D(tie_shape);

   // NDIterator ([pointer to start], [new shape], [stride hack], [current index]);
   NDIterator<string*, shape_type> it_tie(A.data(), tie_shape, tie_stride);

   for(auto ic = D.begin(); ic != D.end(); ++ic, ++it_tie)
   {
      *ic = *it_tie;
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
      cnew_shape[i] = C.shape(permutation[i]);
      cnew_stride[i] = C.stride(permutation[i]);
   }

   // copy A into E
   Tensor<string> E(A);

   // NDIterator ([pointer to start], [new shape], [stride hack], [current index]);
   // get iterator to the first
   NDIterator<string*, shape_type> it_slice_E(E.data()+offset, slice_shape, E.stride());

   // get iterator to the end, i.e. index = { cnew_shape[0], 0, 0 }
   shape_type index_last = slice_shape;
   fill(index_last.begin()+1, index_last.end(), 0);
   NDIterator<string*, shape_type> it_slice_E_end(E.data()+offset, slice_shape, E.stride(), index_last);

   // get iterator to permute within slice (using NDIterator of NDIterator)
   NDIterator<NDIterator<string*, shape_type>, shape_type> it_slice_permt_A(NDIterator<string*, shape_type>(A.data()+offset, slice_shape, A.stride()), cnew_shape, cnew_stride);

   for(; it_slice_E != it_slice_E_end; ++it_slice_E, ++it_slice_permt_A)
   {
      *it_slice_E = *it_slice_permt_A;
   }

   cout.precision(2);
   cout << "Printing E: "; print(E);

   return 0;
}
