#include "test.h"
#include "btas/tensor.h"
#include "btas/generic/contract.h"

using std::cout;
using std::endl;

using btas::Range;

using DTensor = btas::Tensor<double>;

static
std::ostream& 
operator<<(std::ostream& s, const DTensor& X)
    {
    for(auto i : X.range()) s << i << " " << X(i) << "\n";
    return s;
    }

// Set the elements of a Tensor T such that
// T(i,j,k) = 1ijk
// assuming individual dimensions are all less than 10.
// (The 1 in front is just a placeholder.)
void static
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


TEST_CASE("Tensor Contract")
    {

    DTensor T2(3,2);
    fillEls(T2);

    DTensor T3(3,2,4);
    fillEls(T3);

    SECTION("Matrix-like")
        {
        //cout << "T2 = " << T2 << endl;
        enum {i,j,k};

        DTensor R;
        contract(1.0,T2,{j,i},T2,{j,k},0.0,R,{i,k});

        // Correct entries of R should be:
        //
        // 36500 36830
        // 36830 37163
        //

        //const size_t rmax = T2.extent(1),
        //             cmax = T2.extent(1);
        //for(size_t r = 0; r < rmax; ++r)
        //for(size_t c = 0; c < cmax; ++c)
        //    {
        //    double val = 0;
        //    for(size_t i = 0; i < T2.extent(0); ++i)
        //        {
        //        val += T2(i,r)*T2(i,c);
        //        }
        //    CHECK(val == R(r,c));
        //    //cout << r << " " << c << " " << val << " " << R(r,c) << endl;
        //    }
        }


    }
