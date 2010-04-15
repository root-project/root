# MACRO (REFLEX_GENERATE_DICTIONARIES outfiles
#        HEADERS f1 ... fN [SELECTION selection] [OPTIONS o1 ... oN])
#
#   Create Reflex dictionary files from a list of C++ header files.
#   Options and a selection file may be given to genreflex.
#
# MACRO (REFLEX_GENERATE_DICTIONARY infile outfile)
#
#   Creates a rule to run genreflex on infile and create outfile.
#   Use this if for some reason REFLEX_GENERATE_DICTIONARIES() isn't
#   appropriate, e.g. because you need a custom filename for the output
#   file or something similar.
#
# MACRO (REFLEX_ADD_TEST name target DICTIONARIES d1 ... dN)
#
#   Adds a unit test, that is executed when running make test. It will
#   be built with RPATH poiting to the build dir. The test will run a
#   shell script that adds the specified dictionaries into the library
#   lookup path since they are built as modules and can't be linked against.


MACRO (_REFLEX_GET_GENREFLEX_INCLUDES _genreflex_includes)

   # extract the include dirs for reuse by genreflex
   SET(${_genreflex_includes})
   GET_DIRECTORY_PROPERTY(_cmake_includes INCLUDE_DIRECTORIES)

   FOREACH (_arg ${_cmake_includes})
      SET(${_genreflex_includes} ${${_genreflex_includes}} "-I${_arg}")
   ENDFOREACH (_arg ${_cmake_includes})

ENDMACRO (_REFLEX_GET_GENREFLEX_INCLUDES _genreflex_includes)


MACRO (_REFLEX_GET_GENREFLEX_DEFINITIONS _genreflex_definitions)

   SET(${_genreflex_definitions})
   GET_DIRECTORY_PROPERTY(_cmake_definitions COMPILE_DEFINITIONS)

   FOREACH (_arg ${_cmake_definitions})
      SET(${_genreflex_definitions} ${${_genreflex_definitions}} "-D${_arg}")
   ENDFOREACH (_arg ${_cmake_definitions})

ENDMACRO (_REFLEX_GET_GENREFLEX_DEFINITIONS _genreflex_definitions)


MACRO (REFLEX_ASSERT_FILE file type)

   IF (NOT EXISTS ${file})
      MESSAGE(FATAL_ERROR "Cannot find ${type} ${file}")
   ENDIF (NOT EXISTS ${file})

   IF (IS_DIRECTORY ${file})
      MESSAGE(FATAL_ERROR "${file} is a not a file")
   ENDIF (IS_DIRECTORY ${file})

ENDMACRO (REFLEX_ASSERT_FILE file type)


MACRO (_REFLEX_ADD_GENREFLEX_COMMAND infile outfile genreflex_includes genreflex_definitions genreflex_selection genreflex_options)

   REFLEX_ASSERT_FILE(${infile} "header file")

   GET_FILENAME_COMPONENT(_out_dir ${outfile} PATH)
   FILE(RELATIVE_PATH _out_rel ${CMAKE_BINARY_DIR} ${outfile})
   SET(_genreflex_options ${genreflex_options})

   # create the selection option if provided
   IF (NOT "${genreflex_selection}" STREQUAL "")
      GET_FILENAME_COMPONENT(genreflex_selection "${genreflex_selection}" ABSOLUTE)
      REFLEX_ASSERT_FILE(${genreflex_selection} "selection file")
      LIST(APPEND _genreflex_options -s "${genreflex_selection}")
   ENDIF (NOT "${genreflex_selection}" STREQUAL "")

   # link src to target through a genreflex command
   ADD_CUSTOM_COMMAND(OUTPUT ${outfile}
                      COMMAND ${PYTHON_EXECUTABLE}
                      ARGS "${GENREFLEX_SCRIPT}" "${infile}" -o "${outfile}" ${_genreflex_options} ${genreflex_includes} ${genreflex_definitions}
                      IMPLICIT_DEPENDS CXX ${infile}
                      DEPENDS genreflex ${infile} ${genreflex_selection}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      COMMENT "Generating Reflex dictionary ${_out_rel}"
                      VERBATIM)

   MACRO_ADDITIONAL_CLEAN_FILES(${outfile})

   # mark reflex dictionary files as generated
   SET_SOURCE_FILES_PROPERTIES(${outfile} PROPERTIES GENERATED 1)

ENDMACRO (_REFLEX_ADD_GENREFLEX_COMMAND infile outfile genreflex_includes genreflex_definitions genreflex_selection genreflex_options)


MACRO (_REFLEX_EXTRACT_GENREFLEX_OPTIONS _genreflex_files _genreflex_selection _genreflex_options)

   MACRO_PARSE_ARGUMENTS(EXTRACT_OPTIONS "HEADERS;SELECTION;OPTIONS" "" "${ARGN}")

   SET(${_genreflex_files} ${EXTRACT_OPTIONS_HEADERS})
   SET(${_genreflex_options} ${EXTRACT_OPTIONS_OPTIONS})
   SET(${_genreflex_selection} ${EXTRACT_OPTIONS_SELECTION})

ENDMACRO (_REFLEX_EXTRACT_GENREFLEX_OPTIONS _genreflex_files _genreflex_selection _genreflex_options)


MACRO (REFLEX_GENERATE_DICTIONARY infile outfile)

   # extract the includes and defintions for reuse by genreflex
   GET_GENREFLEX_INCLUDES(genreflex_includes)
   GET_GENREFLEX_DEFINITIONS(genreflex_definitions)

   GET_FILENAME_COMPONENT(abs_infile ${infile} ABSOLUTE)
   _REFLEX_ADD_GENREFLEX_COMMAND(${abs_infile} ${outfile} "${genreflex_includes}" "${genreflex_definitions}" "" "")

ENDMACRO (REFLEX_GENERATE_DICTIONARY infile outfile)


MACRO (REFLEX_GENERATE_DICTIONARIES outfiles)

   SET(${outfiles} "")

   # extract the includes and defintions for reuse by genreflex
   _REFLEX_GET_GENREFLEX_INCLUDES(genreflex_includes)
   _REFLEX_GET_GENREFLEX_DEFINITIONS(genreflex_definitions)

   # separate the options and selection file from the input files
   _REFLEX_EXTRACT_GENREFLEX_OPTIONS(genreflex_files genreflex_selection genreflex_options ${ARGN})

   # resolve the selection option
   IF (NOT "${genreflex_selection}" STREQUAL "")
      GET_FILENAME_COMPONENT(genreflex_selection "${genreflex_selection}" ABSOLUTE)
   ENDIF (NOT "${genreflex_selection}" STREQUAL "")

   #CONFIGURE_FILE(${REFLEX_TEMPLATE_DIR}/genreflex.files.in ${_genreflex_source}.files)

   FOREACH (it ${genreflex_files})
      GET_FILENAME_COMPONENT(_abs_file ${it} ABSOLUTE)
      MACRO_MAKE_OUTPUT_FILE(${_abs_file} "" _rflx cpp outfile)
      _REFLEX_ADD_GENREFLEX_COMMAND(${_abs_file} ${outfile} "${genreflex_includes}" "${genreflex_definitions}" "${genreflex_selection}" "${genreflex_options}")
      SET(${outfiles} ${${outfiles}} ${outfile})
   ENDFOREACH (it ${genreflex_files})

ENDMACRO (REFLEX_GENERATE_DICTIONARIES outfiles)


MACRO (_REFLEX_GET_DICTIONARIES_PATH dictionaries _path)

   FOREACH (_d ${dictionaries})

      GET_TARGET_PROPERTY(_type ${_d} TYPE)

      IF (${_type} STREQUAL "MODULE_LIBRARY")

         MACRO_GET_TARGET_DIRECTORY(${_d} _location)
         IF (UNIX)
            SET(${_path} "${_location}:${${_path}}")
         ELSE (UNIX)
            SET(${_path} "${_location}\;${${_path}}")
         ENDIF (UNIX)

      ENDIF (${_type} STREQUAL "MODULE_LIBRARY")

   ENDFOREACH (_d ${dictionaries})

ENDMACRO (_REFLEX_GET_DICTIONARIES_PATH dictionaries _path)


MACRO (GET_TEST_SCOPED_NAME _name _scoped_name)

   SET(_test_qname)
   FILE(RELATIVE_PATH _test_qname ${REFLEX_TEST_DIR} ${CMAKE_PARENT_LIST_FILE})
   GET_FILENAME_COMPONENT(_test_qname ${_test_qname} PATH)
   FILE(TO_CMAKE_PATH ${_test_qname} _test_qname)

   IF (NOT "${_name}" STREQUAL "")
      SET(_test_qname "${_test_qname}/${_name}")
   ENDIF (NOT "${_name}" STREQUAL "")

   SET(${_scoped_name} ${_test_qname})

ENDMACRO (GET_TEST_SCOPED_NAME _name _scoped_name)


MACRO (REFLEX_ADD_SCOPED_TEST _name)

   GET_TEST_SCOPED_NAME("${_name}" _scoped_name)
   SET(_test_key "REGISTERED_TESTS_${_scoped_name}")

   # check for duplicate tests
   IF (DEFINED ${_test_key})
      MESSAGE(FATAL_ERROR "Test ${_scoped_name} was already added")
   ENDIF (DEFINED ${_test_key})

   # register and add the test to ctest
   SET(${_test_key} "1")
   ADD_TEST("${_scoped_name}" ${ARGN})

ENDMACRO (REFLEX_ADD_SCOPED_TEST _name)


MACRO (REFLEX_ADD_LIBRARY name)

   MACRO_PARSE_ARGUMENTS(ADD_LIBRARY "" "TEST" "${ARGN}")

   SET(_add_library_params)

   IF (ADD_LIBRARY_TEST AND NOT REFLEX_BUILD_TESTS)
      SET(_add_library_params ${_add_library_params} EXCLUDE_FROM_ALL)
   ENDIF (ADD_LIBRARY_TEST AND NOT REFLEX_BUILD_TESTS)

   ADD_LIBRARY(${name} ${_add_library_params} ${ADD_LIBRARY_DEFAULT_ARGS})

   IF (ADD_LIBRARY_TEST AND NOT REFLEX_BUILD_TESTS)
      ADD_DEPENDENCIES(build_tests ${name})
   ENDIF (ADD_LIBRARY_TEST AND NOT REFLEX_BUILD_TESTS)

ENDMACRO (REFLEX_ADD_LIBRARY name)


MACRO (REFLEX_ADD_DICTIONARY name)

   MACRO_PARSE_ARGUMENTS(ADD_DICTIONARY "SELECTION;OPTIONS" "WITH_PREFIX;TEST" "${ARGN}")

   SET(_genreflex_files ${ADD_DICTIONARY_DEFAULT_ARGS})
   SET(_genreflex_options ${ADD_DICTIONARY_OPTIONS})
   SET(_genreflex_selection ${ADD_DICTIONARY_SELECTION})

   REFLEX_GENERATE_DICTIONARIES(_dict_files
                                HEADERS ${_genreflex_files}
                                SELECTION ${_genreflex_selection}
                                OPTIONS ${_genreflex_options})

   SET(_add_library_params MODULE)

   IF (ADD_DICTIONARY_TEST)
      SET(_add_library_params ${_add_library_params} TEST)
   ENDIF (ADD_DICTIONARY_TEST)

   REFLEX_ADD_LIBRARY(${name} ${_add_library_params} ${_dict_files})
   TARGET_LINK_LIBRARIES(${name} Reflex)

   IF (ADD_DICTIONARY_WITH_PREFIX)
      SET_TARGET_PROPERTIES(${name} PROPERTIES PREFIX "lib")
   ENDIF (ADD_DICTIONARY_WITH_PREFIX)

ENDMACRO (REFLEX_ADD_DICTIONARY name)


MACRO (REFLEX_ADD_EXECUTABLE name)

   MACRO_PARSE_ARGUMENTS(ADD_EXECUTABLE "" "TEST" "${ARGN}")

   SET(_add_executable_params)

   IF (ADD_EXECUTABLE_TEST AND NOT REFLEX_BUILD_TESTS)
      SET(_add_executable_params ${_add_executable_params} EXCLUDE_FROM_ALL)
   ENDIF (ADD_EXECUTABLE_TEST AND NOT REFLEX_BUILD_TESTS)

   ADD_EXECUTABLE(${name} ${_add_executable_params} ${ADD_EXECUTABLE_DEFAULT_ARGS})

   IF (ADD_EXECUTABLE_TEST AND NOT REFLEX_BUILD_TESTS)
      ADD_DEPENDENCIES(build_tests ${name})
   ENDIF (ADD_EXECUTABLE_TEST AND NOT REFLEX_BUILD_TESTS)

ENDMACRO (REFLEX_ADD_EXECUTABLE name)


MACRO (REFLEX_ADD_TEST name target)

   MACRO_PARSE_ARGUMENTS(REFLEX_ADD_TEST "DICTIONARIES" "" "${ARGN}")

   IF (UNIX)

      IF (APPLE)
         SET(_library_path_variable "DYLD_LIBRARY_PATH")
      ELSE (APPLE)
         SET(_library_path_variable "LD_LIBRARY_PATH")
      ENDIF (APPLE)

      GET_TARGET_PROPERTY(_executable ${target} LOCATION)
      SET(_shell_wrapper ${_executable}.shell)

      # get the location path of the dictionaries that are modules
      _REFLEX_GET_DICTIONARIES_PATH("${REFLEX_ADD_TEST_DICTIONARIES}" _ld_library_path)

      # use ADD_CUSTOM_TARGET() to have the sh-wrapper generated during build time instead of cmake time
      ADD_CUSTOM_COMMAND(TARGET ${target} POST_BUILD
                         COMMAND ${CMAKE_COMMAND}
                         -D_filename=${_shell_wrapper} -D_library_path_variable=${_library_path_variable}
                         -D_ld_library_path="${_ld_library_path}" -D_executable=${_executable}
                         -P ${REFLEX_MODULE_DIR}/ExecViaShell.cmake)

      MACRO_ADDITIONAL_CLEAN_FILES(${_shell_wrapper})

      # under UNIX, set the property WRAPPER_SCRIPT to the name of the generated shell script
      # so it can be queried and used later on easily
      SET_TARGET_PROPERTIES(${target} PROPERTIES WRAPPER_SCRIPT ${_shell_wrapper})

   ELSE (UNIX)

      GET_TARGET_PROPERTY(_executable ${target} LOCATION)
      SET(_shell_wrapper ${CMAKE_CURRENT_BINARY_DIR}/${target}.bat) # .bat because of rpath handling

      # get the location path of the dictionaries that are modules
      MACRO_GET_TARGET_DIRECTORY(Reflex _reflex_path)
      _REFLEX_GET_DICTIONARIES_PATH("${REFLEX_ADD_TEST_DICTIONARIES}" _ld_library_path)
      SET(_ld_library_path "${_ld_library_path}\;${_reflex_path}")

      # use ADD_CUSTOM_TARGET() to have the batch-file-wrapper generated during build time instead of cmake time
      ADD_CUSTOM_COMMAND(TARGET ${target} POST_BUILD
                         COMMAND ${CMAKE_COMMAND}
                         -D_filename="${_shell_wrapper}"
                         -D_ld_library_path="${_ld_library_path}" -D_executable="${_executable}"
                         -P ${REFLEX_MODULE_DIR}/ExecViaShell.cmake)

      # under windows, set the property WRAPPER_SCRIPT just to the name of the executable
      # maybe later this will change to a generated batch file (for setting the PATH so that the dependencies are found)
      SET_TARGET_PROPERTIES(${target} PROPERTIES WRAPPER_SCRIPT ${_executable})

   ENDIF (UNIX)

   REFLEX_ADD_SCOPED_TEST("${name}" ${_shell_wrapper} ${REFLEX_ADD_TEST_DEFAULT_ARGS})

ENDMACRO (REFLEX_ADD_TEST name target)

INCLUDE(ReflexTestMacros)
