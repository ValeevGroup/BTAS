# -*- mode: cmake -*-

# Limit scope of the search if BOOST_ROOT or BOOST_INCLUDEDIR is provided.
if (BOOST_ROOT OR BOOST_INCLUDEDIR)
  set(Boost_NO_SYSTEM_PATHS TRUE)
endif()

# make sure Boost::boost is available, and look for optional serialization component
if (NOT TARGET Boost::boost OR NOT TARGET Boost::serialization)
  set(Boost_BTAS_DEPS_LIBRARIES serialization)
  # try config first
  # OPTIONAL_COMPONENTS in FindBoost available since 3.11
  cmake_minimum_required(VERSION 3.11.0)
  find_package(Boost CONFIG OPTIONAL_COMPONENTS ${Boost_BTAS_DEPS_LIBRARIES})
  if (NOT TARGET Boost::boost)
    find_package(Boost OPTIONAL_COMPONENTS ${Boost_BTAS_DEPS_LIBRARIES})
    if (TARGET Boost::boost)
      set(Boost_USE_CONFIG FALSE)
    endif(TARGET Boost::boost)
  else()
    set(Boost_USE_CONFIG TRUE)
  endif()
endif (NOT TARGET Boost::boost OR NOT TARGET Boost::serialization)

# if Boost not found, and BTAS_BUILD_DEPS_FROM_SOURCE=ON, use FetchContent to build it
if (NOT TARGET Boost::boost)
  if (BTAS_BUILD_DEPS_FROM_SOURCE)
    include(FindOrFetchBoost)
    set(BTAS_BUILT_BOOST_FROM_SOURCE 1)
  else(BTAS_BUILD_DEPS_FROM_SOURCE)
    message(FATAL_ERROR "Boost is a required prerequisite of BTAS, but not found; install Boost or set BTAS_BUILD_DEPS_FROM_SOURCE=ON to obtain from source")
  endif(BTAS_BUILD_DEPS_FROM_SOURCE)
endif (NOT TARGET Boost::boost)

# make BTAS depend on Boost
set(Boost_LIBRARIES Boost::boost)
if (TARGET Boost::serialization)
  list(APPEND Boost_LIBRARIES Boost::serialization)
  target_compile_definitions(BTAS INTERFACE -DBTAS_HAS_BOOST_SERIALIZATION=1)
endif (TARGET Boost::serialization)
target_link_libraries(BTAS INTERFACE ${Boost_LIBRARIES})

# If building unit tests, perform a compile check with Boost
# this is only possible, though, if did not build Boost from source,
# since only imported targets can be used in CMAKE_REQUIRED_LIBRARIES
if (BTAS_BUILD_UNITTEST AND NOT BTAS_BUILT_BOOST_FROM_SOURCE)
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${Boost_LIBRARIES})
  target_compile_definitions(BTAS INTERFACE -DBTAS_HAS_BOOST_ITERATOR=1 -DBTAS_HAS_BOOST_CONTAINER=1 -DBTAS_TARGET_MAX_INDEX_RANK=${TARGET_MAX_INDEX_RANK})

  set(_btas_boostcheck_source "
    #define BOOST_TEST_MAIN main_tester
    #include <boost/test/included/unit_test.hpp>

    #include <fstream>
    #include <cstdio>
    #include <boost/archive/text_oarchive.hpp>
    #include <boost/archive/text_iarchive.hpp>
    #ifdef BTAS_HAS_BOOST_CONTAINER
    #  include <boost/container/small_vector.hpp>
    #endif

    class A {
      public:
        A() : a_(0) {}
        A(int a) : a_(a) {}
        bool operator==(const A& other) const {
          return a_ == other.a_;
        }
      private:
        int a_;

        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
          ar & a_;
        }
    };

    BOOST_AUTO_TEST_CASE( serialization )
    {
      BOOST_CHECK( true );

      A i(1);
      const char* fname = \"tmp.boost\";
      std::ofstream ofs(fname);
      {
        boost::archive::text_oarchive oa(ofs);
        oa << i;
      }
      {
        std::ifstream ifs(fname);
        boost::archive::text_iarchive ia(ifs);
        A i_restored;
        ia >> i_restored;
        BOOST_CHECK(i == i_restored);
        remove(fname);
      }
    }

    #ifdef BTAS_HAS_BOOST_CONTAINER
    BOOST_AUTO_TEST_CASE( container )
    {
      boost::container::small_vector<int, 1> v;
      BOOST_CHECK_NO_THROW(v.push_back(0));
      BOOST_CHECK_NO_THROW(v.push_back(1));
      BOOST_CHECK(v[0] == 0);
      BOOST_CHECK(v[1] == 1);
    }
    #endif  // BTAS_HAS_BOOST_CONTAINER
    ")
  if (CMAKE_CROSSCOMPILING)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("${_btas_boostcheck_source}" BOOST_COMPILES_AND_RUNS)
  else(CMAKE_CROSSCOMPILING)
    include(CheckCXXSourceRuns)
    check_cxx_source_runs("${_btas_boostcheck_source}" BOOST_COMPILES_AND_RUNS)
  endif(CMAKE_CROSSCOMPILING)

  if (NOT BOOST_COMPILES_AND_RUNS)
    message(STATUS "Boost found at ${BOOST_ROOT}, but could not compile and/or run test program")
    message(WARNING "To obtain usable Boost, use your system package manager (HomeBrew, apt, etc.) OR download at www.boost.org and compile (unpacking alone is not enough)")
    message(WARNING "** !! due to missing Boost.Serialization the corresponding unit tests will be disabled !!")
  endif(BOOST_COMPILES_AND_RUNS)
endif(BTAS_BUILD_UNITTEST AND NOT BTAS_BUILT_BOOST_FROM_SOURCE)
