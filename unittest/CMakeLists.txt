# Create the ta_test executable
set(executable btas_test)
set(btas_test_src_files
        contract_test.cc
        mohndle_test.cc
        range_test.cc
        tensor_cp_test.cc
        tensor_blas_test.cc
        tensor_func_test.cc
        tensor_lapack_test.cc
        tensor_test.cc
        tensorview_test.cc
        zb_range_test.cc
        ztensor_cp_test.cc
        test.cc
        )
# the unit tests check BTAS_ASSERT failures, hence can only be built if they throw
if (BTAS_ASSERT_POLICY STREQUAL BTAS_ASSERT_THROW)
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
endif()

# BTAS_ASSERT policy checks: btas/error.h is compiled against each of the 3
# policies (with NDEBUG defined, to confirm that none of them depends on it),
# regardless of which BTAS_ASSERT_POLICY this build was configured with
# (unlike the unit tests above, which need BTAS_ASSERT_THROW)
foreach(_policy THROW ABORT IGNORE)
  set(_exec btas_assert_policy_${_policy}_test)
  add_executable(${_exec} EXCLUDE_FROM_ALL assert_policy_test.cc)
  # N.B. deliberately not linking BTAS: its INTERFACE definitions carry
  #      BTAS_ASSERT_POLICY, and btas/error.h needs nothing but its own headers
  target_include_directories(${_exec} PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_features(${_exec} PRIVATE cxx_std_20)
  target_compile_definitions(${_exec} PRIVATE
          BTAS_ASSERT_POLICY=BTAS_ASSERT_${_policy} NDEBUG)
  add_test(btas/unit/build/assert/${_policy}
          "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target ${_exec})
  set_tests_properties(btas/unit/build/assert/${_policy}
          PROPERTIES FIXTURES_SETUP BTAS_ASSERT_${_policy}_TEST_EXEC)
  add_test(NAME btas/unit/run/assert/${_policy} COMMAND ${_exec})
  set_tests_properties(btas/unit/run/assert/${_policy}
          PROPERTIES FIXTURES_REQUIRED BTAS_ASSERT_${_policy}_TEST_EXEC)
endforeach()
unset(_exec)
unset(_policy)
