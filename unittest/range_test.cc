#include "test.h"

#include <iostream>

#include "btas/range.h"

using std::cout;
using std::endl;

using btas::Range1;
using btas::Range;
using btas::RangeNd;

TEST_CASE("Range1d")
    {
    Range1 r0;
    CHECK(r0.size() == 0);
    Range1 r1(5);
    CHECK(r1.size() == 5);
    long j=0;
    for(auto i=r1.begin();i!=r1.end();++i,++j){
        CHECK(*i== j);
    }
    Range1 r2(-1,4);
    CHECK(r2.size() == 5);
    j=-1;
    for(auto i=r2.begin();i!=r2.end();++i,++j){
        CHECK(*i== j);
    }
    Range1 r3(-1,7,2);
    CHECK(r3.size() == 4);
    j=-1;
    for(auto i=r3.begin();i!=r3.end();++i,j+=2){
        CHECK(*i== j);
    }
    }

TEST_CASE("Range")
    {
    const auto begin = std::array<long, 3>{{-1,-1,-1}};
    const auto size = std::array<std::size_t, 3>{{3,2,3}};

    SECTION("Default")
        {
        Range r0;
        CHECK(r0.area() == 0);
        }
    SECTION("Initialized by extents")
        {
        Range r1(3,2,3);
        CHECK(r1.area()== 18);
        long j=0;
        for(auto i=r1.begin(); i!=r1.end(); ++i) {
            //FIXME
            //index == tmp can not be compiled.
            //const btas::varray<long> tmp={j/6,(j%6)/3,j%3};
            const auto index= *i;
//            CHECK(index == tmp);
            CHECK(index[0] == j/6);
            CHECK(index[1] == (j%6)/3);
            CHECK(index[2] == j%3);
            CHECK(r1.ordinal(*i)== j);
            j++;
        }
        }
    SECTION("Fixed-rank Range")
        {
        RangeNd<CblasRowMajor, std::array<long, 3> > r3(begin, size);
        CHECK(r3.area()== 48);
        typedef RangeNd<CblasRowMajor, std::array<size_t, 3>> Range3d;
        Range3d x;
        CHECK(x.rank()== 3);
        }
    SECTION("Col-major std::vector-based Range")
        {
        RangeNd<CblasColMajor, std::vector<long> > r4(size);
        CHECK(r4.area()== 18);
        }
    SECTION("Initialized by initializer list")
        {
        Range r5={2,3,4};
        CHECK(r5.area()== 24);
        Range r6({-1, -1, -1}, {2, 3, 4});
        CHECK(r6.area()== 60);
        }
    SECTION("Ordinal")
        {
        Range r5={2,3,4};
        long j=0;
        for(auto i : r5){
            CHECK(r5.ordinal(i)== j);
            j++;
        }
        }
    SECTION("Permute")
        {
        const auto p = std::array<size_t, 3>{{2,0,1}};
        Range r5={2,3,4};
        auto r5p = permute(r5, p);
        CHECK(r5p.area()== 24);
        long j=0;
        for(auto i : r5p){
            // ordinal is still in the old order
            CHECK(r5p.ordinal(i)== i[1]*12+i[2]*4+i[0]);
            CHECK(r5p.ordinal(i)== permute(r5.ordinal(),p)(i) );

            CHECK(i[0]==j/6);
            CHECK(i[1]==(j%6)/3);
            CHECK(i[2]==j%3);
            j++;
        }
        for(auto i: r5p)
            for(auto j: r5)
                if(i[0]==j[2] && i[1]==j[0] && i[2]==j[1])
                    CHECK(r5p.ordinal(i)==r5.ordinal(j));

        }
    SECTION("Diag")
        {
        Range r({1,1,1},{3,3,4});
        for(auto i: diag(r))
            CHECK(diag(r).ordinal(i) == r.ordinal(Range::index_type ({i[0],i[0],i[0]})));
        RangeNd<CblasColMajor> rc({1,1,1},{3,3,4});
        for(auto i : diag(rc))
            CHECK(diag(rc).ordinal(i) == rc.ordinal(RangeNd<CblasColMajor>::index_type ({i[0],i[0],i[0]})));
        }
    SECTION("Group and Flatten")
        {
        Range r({2,3,3});
        auto gr= btas::group(r,0,2);
        CHECK(gr.rank()== 2);
        CHECK(gr.area()== 18);
        for(const auto& i : r){

            //FIXME 
            //In current code, the grouped indices are recalculated to give a new index. The new index begins with 0.
            //I am not sure whether it is necessary.
            //FIXME 
            //r.extent(1) is 3. 
            //But I have to set it 3, rather than use r.extent(1)
            //CHECK(r.ordinal(i) == gr.ordinal(Range::index_type ({(i[0]-r.lobound()[0])*r.extent(1)+(i[1]-r.lobound()[1]),i[2]})));
            CHECK(r.ordinal(i) == gr.ordinal(Range::index_type ({(i[0]-r.lobound()[0])*3+(i[1]-r.lobound()[1]),i[2]})));
        }
        auto fr =btas::flatten(r);
        CHECK(fr.rank() == 1);
        CHECK(fr.area()== 18);
        for(const auto& i : r){
            CHECK(r.ordinal(i) == fr.ordinal(Range::index_type{static_cast<Range::index_type::value_type>(i[0]*r.extent(1)*r.extent(2)+i[1]*r.extent(2)+i[2])}));
        }
        }
    SECTION("TieIndex")
        {
        Range r({0,0,0,0},{3,4,4,4});
        auto tr= btas::tieIndex(r,0,2);
        CHECK(tr.rank()== 3);
        CHECK(tr.area()== 48);
        for(auto i : tr)
            CHECK(tr.ordinal(i) == r.ordinal(Range::index_type {i[0],i[1],i[0],i[2]}));
        }

    }
