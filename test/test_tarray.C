#include <iostream>
#include <string>
#include <sstream>
using namespace std;

#include <btas/btas.h>
#include <btas/tarray.h>
#include <btas/generic/permute.h>
using namespace btas;

int main()
{
  TArray<string, 3> A(3,3,3);

  for(auto i: A.range() ) {
    ostringstream so;
    so << i;
    A(i) = so.str();
  }

  TArray<string, 3> B;
  permute(A, {2,0,1}, B);

  auto itrA = A.begin();
  auto itrB = B.begin();
  while (itrA != A.end() && itrB != B.end()) {
     cout << *itrA << " -> " << *itrB << endl;
     ++itrA;
     ++itrB;
  }
}
