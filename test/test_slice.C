#include <iostream>
using namespace std;

#include <btas/tensorview.h>
using namespace btas;

using DTensor = Tensor<double>;
using DTensorView = TensorView<double>;

// Set the elements of a Tensor T such that
// T(i,j,k) = 1ijk
// assuming individual dimensions are all less than 10.
// (The 1 in front is just a placeholder.)
void
fillEls(DTensor& T)
{
  if(T.rank() == 0) return;
  const double base = pow(10.,T.rank());
  const size_t max_ii = T.rank()-1;
  for(auto I : T.range())
    {
    double &val = T(I);
    val = base;
    for(size_t ii = 0; ii <= max_ii; ++ii)
      {
      val += I[ii]*pow(10.,max_ii-ii);
      }
    }
}

int main()
{
  //////////////////////////////////////////////////////////////////////////////
  // diag tests
  //////////////////////////////////////////////////////////////////////////////
  {
  cout << "\nTesting diag\n" << endl;

  Tensor<double> t0(4,4,4); fillEls(t0);

  auto d0 = diag(t0);
  for(auto el : d0) cout << el << endl;
  cout << endl;

  Tensor<double> t1(3,4,4,4,4); fillEls(t1);
  for(auto el : diag(t1)) cout << el << endl;
  cout << endl;

  Tensor<double> t2;
  for(auto el : diag(t2)) cout << el << endl;
  cout << endl;
  }

  return 0;
}
