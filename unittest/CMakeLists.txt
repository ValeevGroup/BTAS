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
target_include_directories(${executable} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(${executable} BTAS)

# Add targets
add_dependencies(check-btas ${executable})

# Add a test
add_test(${executable} ${PROJECT_BINARY_DIR}/unittest/${executable})
set_tests_properties(${executable} PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/unittest" ENVIRONMENT LD_LIBRARY_PATH=$ENV{LIBRARY_PATH})
