# -*- mode: cmake -*-

# Boost can be discovered by every (sub)package but only the top package can *build* it ...
# in either case must declare the components used by BTAS
set(required_components
        headers           # BTAS
        container         # BTAS
        iterator          # BTAS
        random            # BTAS
)
if (DEFINED Boost_REQUIRED_COMPONENTS)
  list(APPEND Boost_REQUIRED_COMPONENTS ${required_components})
  list(REMOVE_DUPLICATES Boost_REQUIRED_COMPONENTS)
else()
  set(Boost_REQUIRED_COMPONENTS "${required_components}" CACHE STRING "Required components of Boost to discovered or built")
endif()
set(optional_components
        serialization # BTAS
)
if (DEFINED Boost_OPTIONAL_COMPONENTS)
  list(APPEND Boost_OPTIONAL_COMPONENTS ${optional_components})
  list(REMOVE_DUPLICATES Boost_OPTIONAL_COMPONENTS)
else()
  set(Boost_OPTIONAL_COMPONENTS "${optional_components}" CACHE STRING "Optional components of Boost to discovered or built")
endif()

if (BTAS_BUILD_DEPS_FROM_SOURCE AND NOT DEFINED Boost_FETCH_IF_MISSING)
  set(Boost_FETCH_IF_MISSING 1)
endif()
include(${vg_cmake_kit_SOURCE_DIR}/modules/FindOrFetchBoost.cmake)
if (DEFINED Boost_SOURCE_DIR)
  set(BTAS_BUILT_BOOST_FROM_SOURCE 1)
endif()

# make BTAS depend on Boost
set(Boost_LIBRARIES Boost::headers;Boost::random)
if (TARGET Boost::serialization)
  list(APPEND Boost_LIBRARIES Boost::serialization)
  target_compile_definitions(BTAS INTERFACE -DBTAS_HAS_BOOST_SERIALIZATION=1)
  # detect if Boost::serialization has been built from source (as opposed to being IMPORTED)
  get_target_property(_boost_serialization_is_imported Boost::serialization IMPORTED)
  if (NOT _boost_serialization_is_imported)
    set(BTAS_BUILT_BOOST_FROM_SOURCE 1)
  endif(NOT _boost_serialization_is_imported)
endif (TARGET Boost::serialization)
target_link_libraries(BTAS INTERFACE ${Boost_LIBRARIES})
target_compile_definitions(BTAS INTERFACE -DBTAS_HAS_BOOST_ITERATOR=1 -DBTAS_HAS_BOOST_CONTAINER=1 -DBTAS_DEFAULT_TARGET_MAX_INDEX_RANK=${TARGET_MAX_INDEX_RANK})

# If building unit tests, perform a compile check with Boost
# this is only possible, though, if did not build Boost from source,
# since only imported targets can be used in CMAKE_REQUIRED_LIBRARIES
if (BUILD_TESTING AND NOT BTAS_BUILT_BOOST_FROM_SOURCE)
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${Boost_LIBRARIES})

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
  endif(NOT BOOST_COMPILES_AND_RUNS)
endif(BUILD_TESTING AND NOT BTAS_BUILT_BOOST_FROM_SOURCE)
