# -*- mode: cmake -*-

# Limit scope of the search if BOOST_ROOT or BOOST_INCLUDEDIR is provided.
if (BOOST_ROOT OR BOOST_INCLUDEDIR)
  set(Boost_NO_SYSTEM_PATHS TRUE)
endif()
#set(Boost_DEBUG TRUE)

# Check for Boost, unless told otherwise (then must set Boost_FOUND, Boost_INCLUDE_DIRS, Boost_LIBRARIES)
if (NOT SKIP_BOOST_SEARCH)
  find_package(Boost 1.33 REQUIRED COMPONENTS serialization OPTIONAL_COMPONENTS container)
endif()

if (Boost_FOUND)

  # Perform a compile check with Boost
  list(APPEND CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIRS})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${Boost_LIBRARIES})
  if (Boost_CONTAINER_FOUND)
    add_definitions(-DHAVE_BOOST_CONTAINER=1 -DBTAS_TARGET_MAX_INDEX_RANK=${TARGET_MAX_INDEX_RANK})
  endif()

  include(CheckCXXSourceRuns)

  CHECK_CXX_SOURCE_RUNS(
      "
      #define BOOST_TEST_MAIN main_tester
      #include <boost/test/included/unit_test.hpp>
      
      #include <fstream>
      #include <cstdio>
      #include <boost/archive/text_oarchive.hpp>
      #include <boost/archive/text_iarchive.hpp>
      #ifdef HAVE_BOOST_CONTAINER
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
      
      #ifdef HAVE_BOOST_CONTAINER
      BOOST_AUTO_TEST_CASE( container )
      {
        boost::container::small_vector<int, 1> v;
        BOOST_CHECK_NO_THROW(v.push_back(0));
        BOOST_CHECK_NO_THROW(v.push_back(1));
        BOOST_CHECK(v[0] == 0);
        BOOST_CHECK(v[1] == 1);
      }
      #endif  // HAVE_BOOST_CONTAINER
      "  BOOST_COMPILES_AND_RUNS)

  if (NOT BOOST_COMPILES_AND_RUNS)
    message(FATAL_ERROR "Boost found at ${BOOST_ROOT}, but could not compile and/or run test program")
  endif(NOT BOOST_COMPILES_AND_RUNS)
  
else()

  # compiling boost properly is too hard ... ask to come back
  message("** BOOST_ROOT was not explicitly set and Boost serialization library was not found")

endif()

if (NOT Boost_INCLUDE_DIRS)
  message(WARNING "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
  message(WARNING "Boost serialization library not found, set BOOST_ROOT to search in the right place (cmake -DBOOST_ROOT=...); if do not have Boost, download at www.boost.org and compile (unpacking alone is not enough)")
  message(WARNING "** !! unit tests will be disabled !!")
  set(BTAS_BUILD_UNITTEST OFF)
else()
  include_directories(${Boost_INCLUDE_DIRS})
endif()
