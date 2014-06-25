#include "test.h"
#include <random>
#include "btas/tensor.h"

using std::cout;
using std::endl;

using btas::Range;
using btas::Tensor;

using DTensor = Tensor<double>;

double static
rng()
    {
    static std::mt19937 rng(std::time(NULL));
    static auto dist = std::uniform_real_distribution<double>{0., 1.};
    return dist(rng);
    }

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


TEST_CASE("Tensor Constructors")
    {

    SECTION("Default Constructor")
        {
        Tensor<double> T0;
        CHECK(T0.size() == 0);
        CHECK(T0.empty());
        Tensor<int> T1;
        CHECK(T1.size() == 0);
        CHECK(T1.empty());
        Tensor<bool> T2;
        CHECK(T2.size() == 0);
        CHECK(T2.empty());
        Tensor<std::complex<double>> T3;
        CHECK(T3.size() == 0);
        CHECK(T3.empty());
        }

    SECTION("Extent Constructor")
        {
        DTensor T0(2,3,4);
        CHECK(T0.rank() == 3);
        CHECK(T0.extent(0) == 2);
        CHECK(T0.extent(1) == 3);
        CHECK(T0.extent(2) == 4);
        CHECK(T0.size() == 2*3*4);
        CHECK(!T0.empty());

        DTensor T1(6,1,2,7,9);
        CHECK(T1.rank() == 5);
        CHECK(T1.extent(0) == 6);
        CHECK(T1.extent(1) == 1);
        CHECK(T1.extent(2) == 2);
        CHECK(T1.extent(3) == 7);
        CHECK(T1.extent(4) == 9);
        CHECK(T1.size() == 6*1*2*7*9);
        CHECK(!T1.empty());

        Tensor<int> Ti(2,4,5);
        CHECK(Ti.rank() == 3);
        CHECK(Ti.extent(0) == 2);
        CHECK(Ti.extent(1) == 4);
        CHECK(Ti.extent(2) == 5);
        CHECK(Ti.size() == 2*4*5);

        Tensor<std::complex<double>> Tc(3,2,9);
        CHECK(Tc.rank() == 3);
        CHECK(Tc.extent(0) == 3);
        CHECK(Tc.extent(1) == 2);
        CHECK(Tc.extent(2) == 9);
        CHECK(Tc.size() == 3*2*9);
        }

    SECTION("Range Constructor")
        {
        Range r0(2,5,3);
        DTensor T0(r0);
        CHECK(T0.rank() == 3);
        CHECK(T0.extent(0) == 2);
        CHECK(T0.extent(1) == 5);
        CHECK(T0.extent(2) == 3);

        Range r1(2,5,3,9,18,6);
        DTensor T1(r1);
        CHECK(T1.rank() == 6);
        }
    }

TEST_CASE("Tensor")
    {
    DTensor T2(3,2);
    fillEls(T2);

    DTensor T3(3,2,4);
    fillEls(T3);

    SECTION("Fill")
        {
        T3.fill(1.);
        for(auto x : T3) CHECK(x == 1);
        }

    SECTION("Generate")
        {
        std::vector<double> data(T3.size());
        for(auto& x : data) x = rng();

        size_t count = 0;
        T3.generate([&](){ return data[count++]; });

        auto it = T3.cbegin();
        size_t j = 0;
        for(; it != T3.cend(); ++j, ++it)
            {
            CHECK(*it == data[j]);
            }
        }
    }
