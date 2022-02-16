cmake_minimum_required(VERSION 3.14)
project(test-btas)

find_package(BTAS REQUIRED)

add_executable(synopsis synopsis.cxx)
target_link_libraries(synopsis BTAS::BTAS)
