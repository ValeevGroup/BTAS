# import HighFive
include(${PROJECT_SOURCE_DIR}/cmake/modules/FindOrFetchHighFive.cmake)
target_compile_definitions(BTAS INTERFACE -DBTAS_HAS_HighFive=1)