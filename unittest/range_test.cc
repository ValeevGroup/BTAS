#include "test.h"
#include "btas/range.h"

using std::cout;
using std::endl;

using btas::Range;

TEST_CASE("Range")
    {

    SECTION("Default")
        {
        Range r0;
        CHECK(r0.area() == 0);
        }

    }
