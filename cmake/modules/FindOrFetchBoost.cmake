if (NOT TARGET Boost::boost)
  include (FetchContent)
  cmake_minimum_required (VERSION 3.14.0)  # for FetchContent_MakeAvailable

  FetchContent_Declare(
          CMAKEBOOST
          GIT_REPOSITORY      https://github.com/Orphis/boost-cmake
  )
  FetchContent_MakeAvailable(CMAKEBOOST)
  FetchContent_GetProperties(CMAKEBOOST
          SOURCE_DIR CMAKEBOOST_SOURCE_DIR
          BINARY_DIR CMAKEBOOST_BINARY_DIR
          )

  # current boost-cmake/master does not install boost correctly, so warn that installed BTAS will not be usable
  # boost-cmake/install_rules https://github.com/Orphis/boost-cmake/pull/45 is supposed to fix it but is inactive
  message(WARNING "Building Boost from source makes BTAS unusable from the install location! Install Boost using package manager or manually and reconfigure/reinstall BTAS to fix this")
  install(TARGETS Boost_serialization EXPORT btas COMPONENT boost-libs)
  export(EXPORT btas
      FILE "${PROJECT_BINARY_DIR}/boost-targets.cmake")
  install(EXPORT btas
      FILE "boost-targets.cmake"
      DESTINATION "${BTAS_INSTALL_CMAKEDIR}"
      COMPONENT boost-libs)

endif(NOT TARGET Boost::boost)

# postcond check
if (NOT TARGET Boost::boost)
  message(FATAL_ERROR "FindOrFetchBoost could not make Boost::boost target available")
endif(NOT TARGET Boost::boost)
