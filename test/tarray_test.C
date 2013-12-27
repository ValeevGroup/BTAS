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

  for(auto i: A.range() ) {
    ostringstream so;
    so << i;
    A(i) = so.str();
  }

  TArray<string, 3> B(transpose(A.range(),{2,0,1}));
  {
  auto itrA = A.begin();
  auto itrB = B.begin();
  for (auto i : B.range()) {
     *(itrB + B.range().ordinal(i)) = *itrA;
     ++itrA;
  }
  }

  auto itrA = A.begin();
  auto itrB = B.begin();
  while (itrA != A.end() && itrB != B.end()) {
     cout << *itrA << " -> " << *itrB << endl;
     ++itrA;
     ++itrB;
  }
}
