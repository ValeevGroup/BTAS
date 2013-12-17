#include <iostream>
#include <string>
#include <sstream>
using namespace std;

#include <btas/btas.h>
#include <btas/tarray.h>
using namespace btas;

int main()
{
   TArray<string, 3> A(3,3,3);

   for(int i = 0; i < A.shape(0); ++i) {
      for(int j = 0; j < A.shape(1); ++j) {
         for(int k = 0; k < A.shape(2); ++k) {
            ostringstream so;
            so << i << "," << j << "," << k;
            A(i,j,k) = so.str();
         }
      }
   }

   TArray<string, 3> B;
   permute(A, {2,1,0}, B);

   auto itrA = A.begin();
   auto itrB = B.begin();
   while (itrA != A.end() && itrB != B.end()) {
      cout << *itrA << " -> " << *itrB << endl;
      ++itrA;
      ++itrB;
   }

   return 0;
}
