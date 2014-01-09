#include <iostream>
#include <iomanip>
using namespace std;

#include <btas/btas.h>
#include <btas/tensor.h>
using namespace btas;

int main()
{
   Tensor<double> A(3,3,3);
   Tensor<double> B(3,3,3);

   auto n = 0u;
   for(auto i = 0u; i < A.extent(0); ++i) {
      for(auto j = 0u; j < A.extent(1); ++j) {
         for(auto k = 0u; k < A.extent(2); ++k, ++n) {
            A(i,j,k) = 0.1*n;
            B(i,j,k) = 0.1*n;
         }
      }
   }

   Tensor<double> C;

   // Either works
// enum {I,J,K,L,P};
   const char I = 'i';
   const char J = 'j';
   const char K = 'k';
   const char L = 'l';
   const char P = 'p';

   // index is requred to be "const integral" or "unsigned integral"
   contract(1.0, A, {I,P,J}, B, {K,L,P}, 1.0, C, {L,I,K,J});

   cout << "contract(A, {" << I << "," << P << "," << J << "}, "
        <<          "B, {" << K << "," << L << "," << P << "}, "
        <<          "C, {" << L << "," << I << "," << K << "," << J << "});" << endl;
   cout << "==================================================" << endl;

   for(auto i = 0u; i < C.extent(0); ++i) {
      for(auto j = 0u; j < C.extent(1); ++j) {
         cout << "C(" << i << "," << j << ",*,*)" << endl;
         for(auto k = 0u; k < C.extent(2); ++k) {
            for(auto l = 0u; l < C.extent(3); ++l) {
               cout << fixed << setprecision(1) << setw(6) << C(i,j,k,l);
            }
            cout << endl;
         }
      }
   }

   return 0;
}
