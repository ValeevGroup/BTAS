# Create the ta_test executable
set(executable btas_test)
set(btas_test_src_files
  tensor_test.cc
  contract_test.cc
  tensor_cp_test.cc
  range_test.cc
  tensor_blas_test.cc
  tensor_func_test.cc
  tensorview_test.cc
  test.cc
)
add_executable(${executable} EXCLUDE_FROM_ALL ${btas_test_src_files})
# Add include directories and compiler flags for ta_test
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Boost_INCLUDE_DIRS})
target_link_libraries(${executable} ${Boost_LIBRARIES} ${LAPACKE_LIBRARIES} ${CBLAS_LIBRARIES})
append_flags(CMAKE_EXE_LINKER_FLAGS "${BLAS_LINKER_FLAGS}")

# Add targets
add_dependencies(${executable} External)
add_dependencies(check ${executable})

# Add a test
add_test(${executable} ${CMAKE_BINARY_DIR}/unittest/${executable} -s) 
set_tests_properties(${executable} PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/unittest" ENVIRONMENT LD_LIBRARY_PATH=$ENV{LIBRARY_PATH})
