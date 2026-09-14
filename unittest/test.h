#ifndef __BTAS_UNITTEST_TEST_H
#define __BTAS_UNITTEST_TEST_H

#include "catch.hpp"

#include <btas/error.h>

// BTAS_ASSERT failures can only be checked if BTAS_ASSERT throws
#if BTAS_ASSERT_POLICY != BTAS_ASSERT_THROW
#  error "unit tests require BTAS_ASSERT to throw, configure with the BTAS_ASSERT_POLICY cmake cache variable set to BTAS_ASSERT_THROW (e.g. by adding -DBTAS_ASSERT_POLICY=BTAS_ASSERT_THROW to cmake command arguments)"
#endif

#endif
