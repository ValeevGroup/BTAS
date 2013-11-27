#include <iostream>
#include <iomanip>
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

int main()
{
   typedef Tensor<double>::shape_type shape_type;

   mt19937 rgen;
   uniform_real_distribution<double> dist(-1.0, 1.0);

   Tensor<double> A(3,4,5); A.generate(bind(dist, rgen));

   cout.precision(2);
   cout << "Printing A: "; print(A);

   // permutation in terms of NDIterator
   shape_type permutation = { 2, 1, 0 };

   shape_type new_shape (A.rank());
   shape_type new_stride(A.rank());
   for(size_t i = 0; i < A.rank(); ++i)
   {
      new_shape[i] = A.shape(permutation[i]);
      new_stride[i] = A.stride(permutation[i]);
   }

   Tensor<double> B(new_shape);
   NDIterator<double*, shape_type> it_permt(A.data(), new_shape, new_stride);
   for(auto ib = B.begin(); ib != B.end(); ++ib, ++it_permt)
   {
      *ib = *it_permt;
   }

   cout.precision(2);
   cout << "Printing B: "; print(B);

   // slicing in terms of NDIterator
   shape_type lower_bound = { 1, 1, 2 };
   shape_type slice_shape = { 2, 2, 2 };

   size_t offset = 0;
   for(size_t i = 0; i < A.rank(); ++i)
   {
      offset += lower_bound[i]*A.stride(i);
   }

   Tensor<double> C(slice_shape);
   NDIterator<double*, shape_type> it_slice(A.data()+offset, slice_shape, A.stride());
   for(auto ic = C.begin(); ic != C.end(); ++ic, ++it_slice)
   {
      *ic = *it_slice;
   }

   cout.precision(2);
   cout << "Printing C: "; print(C);

   return 0;
}
