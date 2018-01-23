# CMake generated Testfile for 
# Source directory: /Users/karlpierce/software/BTAS/unittest
# Build directory: /Users/karlpierce/software/BTAS/cmake-build-debug/unittest
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(btas_test "/Users/karlpierce/software/BTAS/cmake-build-debug/unittest/btas_test" "-s")
set_tests_properties(btas_test PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2018.1.126/mac/tbb/lib:/opt/intel/compilers_and_libraries_2018.1.126/mac/compiler/lib:/opt/intel/compilers_and_libraries_2018.1.126/mac/mkl/lib" WORKING_DIRECTORY "/Users/karlpierce/software/BTAS/unittest")
