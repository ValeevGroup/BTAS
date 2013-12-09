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

   int n = 0;
   for(int i = 0; i < A.shape(0); ++i) {
      for(int j = 0; j < A.shape(1); ++j) {
         for(int k = 0; k < A.shape(2); ++k, ++n) {
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
   contract(CblasRowMajor, 1.0, A, {I,P,J}, B, {K,L,P}, 1.0, C, {L,I,K,J});

   cout << "contract(A, {" << I << "," << P << "," << J << "}, "
        <<          "B, {" << K << "," << L << "," << P << "}, "
        <<          "C, {" << L << "," << I << "," << K << "," << J << "});" << endl;
   cout << "==================================================" << endl;

   for(int i = 0; i < C.shape(0); ++i) {
      for(int j = 0; j < C.shape(1); ++j) {
         cout << "C(" << i << "," << j << ",*,*)" << endl;
         for(int k = 0; k < C.shape(2); ++k) {
            for(int l = 0; l < C.shape(3); ++l) {
               cout << fixed << setprecision(1) << setw(6) << C(i,j,k,l);
            }
            cout << endl;
         }
      }
   }

   return 0;
}
