#ifndef __BTAS_UNITTEST_TEST_H
#define __BTAS_UNITTEST_TEST_H

#include "catch.hpp"

// BTAS_ASSERT failures can only be checked if BTAS_ASSERT_THROWS is defined
#if not defined(BTAS_ASSERT_THROWS)
#  error "unit tests require BTAS_ASSERT_THROWS to be defined, define BTAS_ASSERT_THROWS cmake option (e.g. by adding -DBTAS_ASSERT_THROWS=ON to cmake command arguments)"
#endif

#endif
