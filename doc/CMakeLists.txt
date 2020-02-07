find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(Doxyfile.in Doxyfile @ONLY IMMEDIATE)
  add_custom_target_subproject(btas html
    COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
    SOURCES ${PROJECT_BINARY_DIR}/doc/Doxyfile)
  add_custom_target_subproject(btas doc DEPENDS html-btas)
endif()
