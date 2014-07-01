#include "test.h"
#include "btas/tensor.h"
#include "btas/tensor_func.h"
#include "btas/generic/contract.h"

using std::cout;
using std::endl;

using namespace btas;

using DTensor = btas::Tensor<double>;
using DTensorView = btas::TensorViewOf<DTensor>;

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
template<typename _Tens>
void static
fillEls(_Tens& T)
    {
    if(T.rank() == 0) return;
    const auto base = pow(10.,T.rank());
    const size_t max_ii = T.rank()-1;
    for(auto I : T.range())
        {
        auto& val = T(I);
        val = base;
        for(size_t ii = 0; ii <= max_ii; ++ii)
            {
            val += I[ii]*pow(10.,max_ii-ii);
            }
        }
    }


TEST_CASE("Tensor and TensorView Functions")
    {

    DTensor T2(3,2);
    fillEls(T2);

    DTensor T3(3,2,4);
    fillEls(T3);


    SECTION("Permute")
        {
        auto permutation = { 2, 1, 0 };

        DTensor pT3 = permute(T3,permutation);
        for(size_t i0 = 0; i0 < T3.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T3.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T3.extent(2); ++i2)
            {
            CHECK(pT3(i2,i1,i0) == T3(i0,i1,i2));
            }

        auto pvT3 = permute(T3,permutation);
        for(size_t i0 = 0; i0 < T3.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T3.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T3.extent(2); ++i2)
            {
            CHECK(pvT3(i2,i1,i0) == T3(i0,i1,i2));
            }
        }


    }
