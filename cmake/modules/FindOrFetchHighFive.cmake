if (NOT TARGET HighFive)
  include (FetchContent)
  cmake_minimum_required (VERSION 3.14.0)  # for FetchContent_MakeAvailable
  find_package(HDF5)
  include_directories(${HDF5_INCLUDE_DIRS})
  target_link_libraries(BTAS INTERFACE ${HDF5_LIBRARIES})

  FetchContent_Declare(
      CMAKEHIGHFIVE
      GIT_REPOSITORY      https://github.com/BlueBrain/HighFive
  )
  set(HIGHFIVE_USE_BOOST OFF)
  FetchContent_MakeAvailable(CMAKEHIGHFIVE)
  FetchContent_GetProperties(CMAKEHIGHFIVE
                             SOURCE_DIR CMAKEHIGHFIVE_SOURCE_DIR
                             BINARY_DIR CMAKEHIGHFIVE_BINARY_DIR
                             )

  target_link_libraries(BTAS INTERFACE HighFive)

endif(NOT TARGET HighFive)

# postcond check
if (NOT TARGET HighFive)
  message(FATAL_ERROR "FindOrFetchHighFive could not make HF::highfive target available")
endif(NOT TARGET HighFive)
