#include "test.h"
#include <random>
#include "btas/tensorview.h"
#include "btas/tensor.h"
#include "btas/tensor_func.h"

using std::cout;
using std::endl;

using btas::Range;
using btas::Tensor;
using DTensor = Tensor<double>;
using btas::TensorView;
using namespace btas;

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

TEST_CASE("Tensor View Constructors")
    {

    SECTION("Constructed from Full Tensors")
        {
        DTensor T0(2,3,4);
        fillEls(T0);
        TensorView<double> T0v(T0);
        CHECK(T0v == T0);
        T0v(0,0,0) = 1.0;
        CHECK(T0v(0,0,0) == 1.0);
        //Cannot make a tensorview with different tensor type in this way.
        //TensorView<float> T0vf(T0);
        //CHECK(T0vf == T0);
        
        auto T0vd = make_view<double>(T0);
        CHECK(T0vd == T0);
        T0vd(0,0,0) = 1.0;
        CHECK(T0vd(0,0,0) == 1.0);

        auto T0cvd = make_cview<double>(T0);
        //auto T0cvd = make_cview(T0);
        CHECK(T0cvd == T0);
        //for(auto i: T0cvd) 
        //     cout << i<<endl;  // This can be used.
        //for( auto i: T0cvd.range())
        //    cout << T0cvd(i)<<endl;
        //cout << T0cvd(0,0,0)<<endl;

        auto T0vf = make_view<float>(T0);
        CHECK(T0vf == T0);
        //cout << T0vf(0,0,0)<<endl; // This cannot be used.
        //T0vf(0,0,0) = 1.0f;
        //CHECK(T0vf(0,0,0) == 1.0);

        }

    SECTION("Constructed from Tensor permute")
        {
        DTensor T0(2,3,4);
        fillEls(T0);
        const auto T0_cref = T0;
        auto prange0 = permute(T0.range(),{2,1,0});

        const auto T0cvr = make_view(prange0, T0_cref.storage());
        //T0cvr(0,0,0) =1.0; // This is not allowed.
        //T0cvr(i,j,k) can be used. 
        for(size_t i0=0; i0< T0.extent(0);i0++)
        for(size_t i1=0; i1< T0.extent(1);i1++)
        for(size_t i2=0; i2< T0.extent(2);i2++)
            CHECK( T0cvr(i2,i1,i0) == T0(i0,i1,i2));

        // read only view
        auto T0cv = make_cview(prange0, T0.storage());
        //FIXME T0cv(0,0,0) cannot be used.
        //cout << T0cv(0,0,0)<<endl;
        //CHECK( T0cv(0,0,0) == T0(0,0,0));
        //for( auto i : T0cv.range())
        //    cout << T0cv(i)<<endl;
        //T0cv(0,0,0)=0.1; // This is not allowed.
        //CHECK(T0cv(0,0,0) == 0.1);
        //for(auto i: T0cv) cout << i<<endl; // This is okay. 
        
        // readwrite view
        auto T0vw = make_view(prange0, T0.storage());
        T0vw(0,0,0)= 1.0;
        CHECK(T0vw(0,0,0) == 1.0);
        CHECK(T0vw(0,0,0) == T0(0,0,0));


        auto T0ncvr = make_view(prange0, T0_cref.storage());
        //T0ncvr(0,0,0) =1.0;
        //FIXME
        //cout << T0ncvr(0,0,0)<<endl; T0ncvr(0,0,0) cannot be used.
        //CHECK(T0ncvr(0,0,0) == 1.0);

        }
        
    }

