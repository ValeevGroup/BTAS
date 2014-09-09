# Create the ta_test executable
set(executable btas_test)
file(GLOB btas_test_src_files *.cc)
add_executable(${executable} EXCLUDE_FROM_ALL ${btas_test_src_files})

# Add include directories and compiler flags for ta_test
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Boost_INCLUDE_DIRS})
target_link_libraries(${executable} ${BTAS_LIBRARIES})

# Add targets
add_dependencies(${executable} External)
add_dependencies(check ${executable})

# Add a test
add_test(${executable} ${executable})
