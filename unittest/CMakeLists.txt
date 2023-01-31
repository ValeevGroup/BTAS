# Create the ta_test executable
set(executable btas_test)
set(btas_test_src_files
        ztensor_cp_test.cc
        tensor_test.cc
        contract_test.cc
        mohndle_test.cc
        tensor_cp_test.cc
        range_test.cc
        tensor_blas_test.cc
        tensor_func_test.cc
        tensor_lapack_test.cc
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

# Add test suite
add_test(btas/unit/build "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target ${executable})
set_tests_properties(btas/unit/build PROPERTIES FIXTURES_SETUP BTAS_UNIT_TESTS_EXEC)
add_test(NAME btas/unit/run
        COMMAND ${executable})
set_tests_properties(btas/unit/run
        PROPERTIES
          FIXTURES_REQUIRED BTAS_UNIT_TESTS_EXEC
          WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/unittest"
        )
