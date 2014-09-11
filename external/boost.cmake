# -*- mode: cmake -*-

# Limit scope of the search if BOOST_ROOT or BOOST_INCLUDEDIR is provided.
if (BOOST_ROOT OR BOOST_INCLUDEDIR)
  set(Boost_NO_SYSTEM_PATHS TRUE)
endif()
  
# Check for Boost
find_package(Boost 1.33 COMPONENTS serialization)

if (Boost_FOUND)

  # Perform a compile check with Boost
  list(APPEND CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIR})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${Boost_LIBRARIES})

  CHECK_CXX_SOURCE_RUNS(
      "
      #define BOOST_TEST_MAIN main_tester
      #include <boost/test/included/unit_test.hpp>
      
      #include <fstream>
      #include <cstdio>
      #include <boost/archive/text_oarchive.hpp>
      #include <boost/archive/text_iarchive.hpp>
      
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

      BOOST_AUTO_TEST_CASE( tester )
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
      "  BOOST_COMPILES_AND_RUNS)

  if (NOT BOOST_COMPILES_AND_RUNS)
    message(FATAL_ERROR "Boost found at ${BOOST_ROOT}, but could not compile and/or run test program")
  endif()
  
elseif(BTAS_EXPERT)

  message("** BOOST was not explicitly set")
  message(FATAL_ERROR "** Downloading and building Boost is explicitly disabled in EXPERT mode")

else()

  include(ExternalProject)
  
  # Set source and build path for Boost in the TiledArray Project
  set(BOOST_DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/external/src)
  set(BOOST_SOURCE_DIR   ${PROJECT_SOURCE_DIR}/external/src/boost)
  set(BOOST_BUILD_DIR   ${PROJECT_BINARY_DIR}/external/build/boost)

  # Set the external source
  if (EXISTS ${PROJECT_SOURCE_DIR}/external/src/boost.tar.gz)
    # Use local file
    set(BOOST_URL ${PROJECT_SOURCE_DIR}/external/src/boost.tar.gz)
    set(BOOST_URL_HASH "")
  else()
    # Downlaod remote file
    set(BOOST_URL
        http://downloads.sourceforge.net/project/boost/boost/1.54.0/boost_1_54_0.tar.gz)
    set(BOOST_URL_HASH MD5=efbfbff5a85a9330951f243d0a46e4b9)
  endif()

  message("** Will build Boost from ${BOOST_URL}")

  ExternalProject_Add(boost
    PREFIX ${CMAKE_INSTALL_PREFIX}
    STAMP_DIR ${BOOST_BUILD_DIR}/stamp
   #--Download step--------------
    URL ${BOOST_URL}
    URL_HASH ${BOOST_URL_HASH}
    DOWNLOAD_DIR ${BOOST_DOWNLOAD_DIR}
   #--Configure step-------------
    SOURCE_DIR ${BOOST_SOURCE_DIR}
    CONFIGURE_COMMAND ""
   #--Build step-----------------
    BUILD_COMMAND ""
   #--Install step---------------
    INSTALL_COMMAND ""
   #--Custom targets-------------
    STEP_TARGETS download
    )

  add_dependencies(External boost)
  install(
    DIRECTORY ${BOOST_SOURCE_DIR}/boost
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT boost
    )
  set(Boost_INCLUDE_DIRS ${BOOST_SOURCE_DIR})
  set(Boost_LIBRARIES "-L${BOOST_BUILD_DIR}/lib -lboost_serialization")

endif()

# Set the  build variables
include_directories(${Boost_INCLUDE_DIRS})
