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

  for(auto i: A.range().dim(0) ) {
    for(auto j: A.range().dim(1) ) {
      for(auto k: A.range().dim(2) ) {
           ostringstream so;
           so << i << "," << j << "," << k;
           A(i,j,k) = so.str();
        }
     }
  }
  std::copy(A.begin(), A.end(), std::ostream_iterator<string>(cout, " "));

#if 0
  TArray<string, 3> B;
  permute(A, {2,1,0}, B);

  auto itrA = A.begin();
  auto itrB = B.begin();
  while (itrA != A.end() && itrB != B.end()) {
     cout << *itrA << " -> " << *itrB << endl;
     ++itrA;
     ++itrB;
  }
#endif
}
