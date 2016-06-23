
INCLUDE(CheckFunctionExists)
INCLUDE(CheckIncludeFile)

# This macro checks for the presence of the combination of libraries
# given by _list.  If the combination is found, this macro checks (using the 
# Check_Function_Exists macro) whether can link against that library
# combination using the name of a routine given by _name using the linker
# flags given by _flags.  If the combination of libraries is found and passes
# the link test, LIBRARIES is set to the list of complete library paths that
# have been found.  Otherwise, LIBRARIES is set to FALSE.
#
# If _search_include is set, also checks for presence of include file given by _include
# using Check_Include_File macro
# 
# N.B. _prefix is the prefix applied to the names of all cached variables that
# are generated internally and marked advanced by this macro.
MACRO(CHECK_LIBRARY_LIST LIBRARIES _prefix _name _flags _list _include _search_include)
  SET(__list)
  FOREACH(_elem ${_list})
    IF(__list)
      SET(__list "${__list} - ${_elem}")
    ELSE(__list)
      SET(__list "${_elem}")
    ENDIF(__list)
  ENDFOREACH(_elem)
  IF(_verbose)
    MESSAGE(STATUS "Checking for [${__list}]")
  ENDIF(_verbose)
  SET(_libraries_work TRUE)
  SET(${LIBRARIES})
  SET(_combined_name)
  SET(_paths)
  FOREACH(_library ${_list})
    SET(_combined_name ${_combined_name}_${_library})
    # did we find all the libraries in the _list until now?
    # (we stop at the first unfound one)
    IF(_libraries_work)      
      IF(APPLE) 
        FIND_LIBRARY(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 ENV 
          DYLD_LIBRARY_PATH 
          )
      ELSE(APPLE)
        FIND_LIBRARY(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/lib/libblas /usr/local/lib64 /usr/lib64 ENV 
          LD_LIBRARY_PATH 
          )
      ENDIF(APPLE)
      MARK_AS_ADVANCED(${_prefix}_${_library}_LIBRARY)
      IF(${_prefix}_${_library}_LIBRARY)
        GET_FILENAME_COMPONENT(_path ${${_prefix}_${_library}_LIBRARY} PATH)
        LIST(APPEND _paths ${_path}/../include ${_path}/../../include)
      ENDIF(${_prefix}_${_library}_LIBRARY)
      SET(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      SET(_libraries_work ${${_prefix}_${_library}_LIBRARY})
      IF(_verbose)
        MESSAGE(STATUS "Searched for library ${_library}: result=${${_prefix}_${_library}_LIBRARY}")
      ENDIF(_verbose)
    ENDIF(_libraries_work)
  ENDFOREACH(_library ${_list})
  # Test include
  SET(_bug_search_include ${_search_include}) #CMAKE BUG!!! SHOULD NOT BE THAT
  IF(_bug_search_include)
    FIND_PATH(${_prefix}${_combined_name}_INCLUDE ${_include} ${_paths})
    MARK_AS_ADVANCED(${_prefix}${_combined_name}_INCLUDE)
    IF(${_prefix}${_combined_name}_INCLUDE)
      IF (_verbose)
        MESSAGE(STATUS "Includes found: ${_prefix}${_combined_name}_INCLUDE=${${_prefix}${_combined_name}_INCLUDE}  ${_prefix}_INCLUDE_FILE=${${_prefix}_INCLUDE_FILE}")
      ENDIF (_verbose)
      SET(${_prefix}_INCLUDE_DIR ${${_prefix}${_combined_name}_INCLUDE})
      SET(${_prefix}_INCLUDE_FILE ${_include})
    ELSE(${_prefix}${_combined_name}_INCLUDE)
      SET(_libraries_work FALSE)
    ENDIF(${_prefix}${_combined_name}_INCLUDE)
  ELSE(_bug_search_include)
    SET(${_prefix}_INCLUDE_DIR)
    SET(${_prefix}_INCLUDE_FILE ${_include})
  ENDIF(_bug_search_include)
  # Test this combination of libraries.
  IF(_libraries_work)
    SET(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}})
    CHECK_FUNCTION_EXISTS(${_name} ${_prefix}${_combined_name}_WORKS)
    SET(CMAKE_REQUIRED_LIBRARIES)
    MARK_AS_ADVANCED(${_prefix}${_combined_name}_WORKS)
    SET(_libraries_work ${${_prefix}${_combined_name}_WORKS})
    IF(_verbose AND _libraries_work)
      MESSAGE(STATUS "Libraries found: ${_prefix}_${_library}_LIBRARY = ${${_prefix}_${_library}_LIBRARY}")
    ENDIF(_verbose AND _libraries_work)
  ENDIF(_libraries_work)
  # Fin
  IF(NOT _libraries_work)
    SET(${LIBRARIES} NOTFOUND)
  ENDIF(NOT _libraries_work)
ENDMACRO(CHECK_LIBRARY_LIST)

