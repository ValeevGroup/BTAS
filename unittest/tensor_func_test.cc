#include "test.h"

#include <iostream>

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
    SECTION("Diag")
        {
        Range r({1,1,1},{3,3,4});
        DTensor T1(r);
        fillEls(T1);
        DTensor Td=diag(T1);
        for(size_t i0 = 0; i0 < T1.extent(0); ++i0)
            CHECK(T1(i0+1,i0+1,i0+1) == Td(i0+1));
        DTensor Tvd=diag(T1);
        for(size_t i0 = 0; i0 < T1.extent(0); ++i0)
            CHECK(T1(i0+1,i0+1,i0+1) == Tvd(i0+1));


        RangeNd<CblasColMajor> rc({1,1,1},{3,3,4});
        DTensor Tc1(rc);
        fillEls(Tc1);
        DTensor Tcd=btas::diag(T1);
        for(size_t i0 = 0; i0 < Tc1.extent(0); ++i0)
            CHECK(Tc1(i0+1,i0+1,i0+1) == Tcd(i0+1));
        auto Tcvd=btas::diag(T1);
        for(size_t i0 = 0; i0 < Tc1.extent(0); ++i0)
            CHECK(Tc1(i0+1,i0+1,i0+1) == Tcvd(i0+1));
        }
    SECTION("Group")
        {
        Range r({2,3,3});
        DTensor T1(r);
        fillEls(T1);
        DTensor Tg=btas::group(T1,0,2);
        CHECK(Tg.rank()== 2);
        CHECK(Tg.size()== 18);
        for(size_t i0 = 0; i0 < T1.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T1.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T1.extent(2); ++i2)
            {
            CHECK(T1(i0,i1,i2)== Tg(i0*T1.extent(1)+i1,i2));
            }

        auto Tvg=btas::group(T1,0,2);
        CHECK(Tvg.rank()== 2);
        CHECK(Tvg.size()== 18);
        for(size_t i0 = 0; i0 < T1.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T1.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T1.extent(2); ++i2)
            {
            CHECK(T1(i0,i1,i2)== Tvg(i0*T1.extent(1)+i1,i2));
            }
        }
    SECTION("Flatten")
        {
        Range r({2,3,3});
        DTensor T1(r);
        fillEls(T1);
        DTensor Tg=btas::flatten(T1);
        CHECK(Tg.rank()== 1);
        CHECK(Tg.size()== 18);
        for(size_t i0 = 0; i0 < T1.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T1.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T1.extent(2); ++i2)
            {
            CHECK(T1(i0,i1,i2)== Tg(i0*T1.extent(2)*T1.extent(1)+i1*T1.extent(2)+i2));
            CHECK(T1(i0+T1.range().lobound()[0],i1+T1.range().lobound()[1],i2+T1.range().lobound()[2])== Tg(i0*T1.extent(2)*T1.extent(1)+i1*T1.extent(2)+i2));
            }

        auto Tvg=btas::flatten(T1);
        CHECK(Tvg.rank()== 1);
        CHECK(Tvg.size()== 18);
        for(size_t i0 = 0; i0 < T1.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T1.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T1.extent(2); ++i2)
            {
            CHECK(T1(i0,i1,i2)== Tvg(i0*T1.extent(2)*T1.extent(1)+i1*T1.extent(2)+i2));
            CHECK(T1(i0+T1.range().lobound()[0],i1+T1.range().lobound()[1],i2+T1.range().lobound()[2])== Tvg(i0*T1.extent(2)*T1.extent(1)+i1*T1.extent(2)+i2));
            }
        }
    SECTION("Flatten Col Major")
        {
        RangeNd<CblasColMajor> rc({1,1,1},{3,3,4});
        DTensor T1(rc);
        fillEls(T1);
        DTensor Tg=btas::flatten(T1);
        CHECK(Tg.rank()== 1);
        CHECK(Tg.size()== 12);
        for(size_t i0 = 0; i0 < T1.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T1.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T1.extent(2); ++i2)
            //After Flatten, the 1st order range begin with 0.
            CHECK(T1(i0+T1.range().lobound()[0],i1+T1.range().lobound()[1],i2+T1.range().lobound()[2])== Tg(i0*T1.extent(2)*T1.extent(1)+i1*T1.extent(2)+i2));

        auto Tvg=btas::flatten(T1);
        CHECK(Tvg.rank()== 1);
        CHECK(Tvg.size()== 12);
        for(size_t i0 = 0; i0 < T1.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T1.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T1.extent(2); ++i2)
            //After Flatten, the 1st order range begin with 0.
            CHECK(T1(i0+T1.range().lobound()[0],i1+T1.range().lobound()[1],i2+T1.range().lobound()[2])== Tvg(i0*T1.extent(2)*T1.extent(1)+i1*T1.extent(2)+i2));

        }

    SECTION("TieIndex")
        {
        DTensor T(3,4,4,4);
        Range r({0,0,0,0},{3,4,4,4});
        DTensor tT= btas::tieIndex(T,0,2); 
        CHECK(tT.rank()== 3);
        CHECK(tT.size()== 48);
        for(size_t i0 = 0; i0 < T.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T.extent(3); ++i2)
            CHECK(tT(i0,i1,i2) == T(i0,i1,i0,i2));

        auto tTv= btas::tieIndex(T,0,2); 
        CHECK(tTv.rank()== 3);
        CHECK(tTv.size()== 48);
        for(size_t i0 = 0; i0 < T.extent(0); ++i0)
        for(size_t i1 = 0; i1 < T.extent(1); ++i1)
        for(size_t i2 = 0; i2 < T.extent(3); ++i2)
            CHECK(tTv(i0,i1,i2) == T(i0,i1,i0,i2));
        }


    }
