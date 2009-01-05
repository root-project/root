MACRO(MACRO_SCRUB_DIR path)

   FILE(GLOB _sub_dirs ${path}/*)
   LIST(LENGTH _sub_dirs _sub_dirs_length)

   IF (${_sub_dirs_length} GREATER 0)
      FILE(REMOVE_RECURSE ${_sub_dirs})
   ENDIF (${_sub_dirs_length} GREATER 0)

ENDMACRO(MACRO_SCRUB_DIR path)


MACRO (REFLEX_ADD_MACRO_TEST _name)

   IF (NOT REFLEX_TESTING)

      GET_TEST_SCOPED_NAME(${_name} _scoped_name)
      SET(_test_source_dir "${Reflex_BINARY_DIR}/Testing/Temporary/MacroTests/${_scoped_name}")
      SET(_test_binary_dir "${_test_source_dir}")
      SET(_script_file_name "${_test_source_dir}/CMakeLists.txt")

      SET(_test_output_dir "${CMAKE_BINARY_DIR}/Testing/Temporary/Tests")
      IF (NOT EXISTS ${_test_output_dir})
         FILE(MAKE_DIRECTORY ${_test_output_dir})
      ENDIF (NOT EXISTS ${_test_output_dir})

      IF (NOT EXISTS ${_script_file_name})

         MESSAGE(STATUS "Writing out project files for ${_scoped_name}")

         FILE(MAKE_DIRECTORY ${_test_source_dir})

         FILE(WRITE ${_script_file_name} "INCLUDE(MacroLibrary)\n")
         FILE(APPEND ${_script_file_name} "INCLUDE(ReflexMacros)\n")
         FILE(APPEND ${_script_file_name} "INCLUDE(${CMAKE_CURRENT_LIST_FILE})\n")
         FILE(APPEND ${_script_file_name} "\n")
         FILE(APPEND ${_script_file_name} "MACRO_SCRUB_DIR(\"${_test_output_dir}\")\n")
         FILE(APPEND ${_script_file_name} "${_name}()")

      ENDIF (NOT EXISTS ${_script_file_name})

      REFLEX_ADD_SCOPED_TEST("${_name}" ${CMAKE_COMMAND}
                             "-DCMAKE_MODULE_PATH:PATH=${CMAKE_MODULE_PATH}"
                             "-DCMAKE_PROJECT_NAME:STRING=${_name}"
                             "-DCMAKE_VERSION:STRING=${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}"
                             "-DREFLEX_TESTING:BOOLEAN=TRUE"
                             "-DREFLEX_TEST_OUTPUT_DIR:PATH=${_test_output_dir}"
                             "-DREFLEX_INCLUDE_DIR:PATH=${REFLEX_INCLUDE_DIR}"
                             "-DREFLEX_LIBRARY:PATH=${REFLEX_LIBRARY}"
                             "-DCPPUNIT_INCLUDE_DIR:PATH=${CPPUNIT_INCLUDE_DIR}"
                             "-DCPPUNIT_LIBRARY:PATH=${CPPUNIT_LIBRARY}"
                             "-DPYTHON_EXECUTABLE:PATH=${PYTHON_EXECUTABLE}"
                             "-DGCCXML:PATH=${GCCXML}"
                             "-DGENREFLEX_SCRIPT:PATH=${GENREFLEX_SCRIPT}"
                             -P ${_script_file_name})

    ENDIF (NOT REFLEX_TESTING)

ENDMACRO (REFLEX_ADD_MACRO_TEST _name)

MACRO (_GET_TEST_ARGS _list _prefix _match_type _expected)

   LIST(LENGTH ${_list} _list_length)

   IF (NOT _list_length EQUAL 2)
      MESSAGE(FATAL_ERROR "${_prefix} called with incorrect number of arguments")
   ENDIF (NOT _list_length EQUAL 2)

   LIST(GET ${_list} 0 ${_match_type})
   LIST(GET ${_list} 1 ${_expected})

ENDMACRO (_GET_TEST_ARGS _list _prefix _match_type _expected)

MACRO (REFLEX_ASSERT_GENREFLEX_CLI)

   MACRO_PARSE_ARGUMENTS(RUN_GENREFLEX_TEST "ARGS;RESULT;STDOUT;STDERR" "" "${ARGN}")

   SET(_result_match_type)
   SET(_expected_result)
   _GET_TEST_ARGS(RUN_GENREFLEX_TEST_RESULT RESULT _result_match_type _expected_result)

   SET(_out_match_type STREQUAL)
   SET(_expected_out "")
   _GET_TEST_ARGS(RUN_GENREFLEX_TEST_STDOUT STDOUT _out_match_type _expected_out)

   SET(_err_match_type STREQUAL)
   SET(_expected_err "")
   _GET_TEST_ARGS(RUN_GENREFLEX_TEST_STDERR STDERR _err_match_type _expected_err)

   SET(_result -1)
   SET(_out "")
   SET(_err "")

   MESSAGE(STATUS "Running ${PYTHON_EXECUTABLE} ${GENREFLEX_SCRIPT};${RUN_GENREFLEX_TEST_ARGS}")

   SET(_out_file ${REFLEX_TEST_OUTPUT_DIR}/out.txt)
   SET(_err_file ${REFLEX_TEST_OUTPUT_DIR}/err.txt)

   EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ${GENREFLEX_SCRIPT};${RUN_GENREFLEX_TEST_ARGS}
                   RESULT_VARIABLE _result
                   OUTPUT_FILE ${_out_file}
                   ERROR_FILE ${_err_file}
                   WORKING_DIRECTORY ${REFLEX_TEST_OUTPUT_DIR})

   FILE(READ ${_out_file} _out)
   FILE(READ ${_err_file} _err)

   MACRO_ASSERT("Unexpected result for invocation with arguments [${RUN_GENREFLEX_TEST_ARGS}]" ${_result} EQUAL ${_expected_result})
   MACRO_ASSERT("Unexpected standard output for invocation with arguments [${RUN_GENREFLEX_TEST_ARGS}]" "${_out}" "${_out_match_type}" "${_expected_out}")
   MACRO_ASSERT("Unexpected standard error for invocation with arguments [${RUN_GENREFLEX_TEST_ARGS}]" "${_err}" "${_err_match_type}" "${_expected_err}")

ENDMACRO (REFLEX_ASSERT_GENREFLEX_CLI)

MACRO (MACRO_RESOLVE_TEST_FILE _name _absolute_name)

   SET(${_absolute_name} ${_name})
   IF (NOT IS_ABSOLUTE ${_name})
      SET(${_absolute_name} ${REFLEX_TEST_OUTPUT_DIR}/${_name})
   ENDIF (NOT IS_ABSOLUTE ${_name})

ENDMACRO (MACRO_RESOLVE_TEST_FILE _name _absolute_name)

MACRO (MACRO_WRITE_TEST_FILE _name _content)

   MACRO_RESOLVE_TEST_FILE(${_name} _absolute_name)
   FILE(WRITE ${_absolute_name} ${_content})

ENDMACRO (MACRO_WRITE_TEST_FILE _name _content)

MACRO (MACRO_ASSERT _message _actual _match_type _expected)

   IF (NOT "${_actual}" ${_match_type} "${_expected}")
      MESSAGE(FATAL_ERROR "${_message}:\nExpected (${_match_type}):\n${_expected}\nActual:\n${_actual}")
   ENDIF (NOT "${_actual}" ${_match_type} "${_expected}")

ENDMACRO (MACRO_ASSERT _message _actual _match_type _expected)

MACRO (MACRO_ASSERT_TEST_FILE_EXISTS _file)

   SET(_message "Expected file: ")

   MACRO_RESOLVE_TEST_FILE(${_file} _absolute_file)
   IF (NOT EXISTS ${_absolute_file})
      MESSAGE(FATAL_ERROR "${_message}: ${_absolute_file}")
   ENDIF (NOT EXISTS ${_absolute_file})

ENDMACRO (MACRO_ASSERT_TEST_FILE_EXISTS _file)

MACRO (MACRO_ASSERT_TEST_FILE_NOT_EXISTS _file)

   SET(_message "Unexpected file: ")

   MACRO_RESOLVE_TEST_FILE(${_file} _absolute_file)
   IF (EXISTS ${_absolute_file})
      MESSAGE(FATAL_ERROR "${_message}: ${_absolute_file}")
   ENDIF (EXISTS ${_absolute_file})

ENDMACRO (MACRO_ASSERT_TEST_FILE_NOT_EXISTS _file)

MACRO (REFLEX_ADD_SINGLE_TEST _name)

   # create the scoped target names
   GET_TEST_SCOPED_NAME(${_name} _qname)
   STRING(REPLACE "/" "_" _qname ${_qname})

   MACRO_PARSE_ARGUMENTS(_TEST "HEADERS;TEST" "" "${ARGN}")

   # create the dictionary
   SET(_dict_target ${_qname}_rflx)
   REFLEX_ADD_DICTIONARY(${_dict_target} TEST ${_TEST_HEADERS}
                         OPTIONS --quiet)

   # TODO: needs to become a parameter or move to parent scope
   SET(UTIL_INCLUDE_DIR ${REFLEX_TEST_DIR})
   INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR} ${UTIL_INCLUDE_DIR} ${REFLEX_INCLUDE_DIR} ${CPPUNIT_INCLUDE_DIR})

   # create the test executable
   SET(_test_target ${_qname})
   REFLEX_ADD_EXECUTABLE(${_test_target} TEST ${_TEST_TEST})
   TARGET_LINK_LIBRARIES(${_test_target} Reflex ${DL_LIBRARY} ${CPPUNIT_LIBRARY})

   # determine the dictionary path relative to the test executable
   GET_TARGET_PROPERTY(_dict_lib ${_dict_target} LOCATION)
   GET_TARGET_PROPERTY(_test_exe ${_test_target} LOCATION)
   GET_FILENAME_COMPONENT(_test_exe_dir ${_test_exe} PATH)
   FILE(RELATIVE_PATH _dict_lib_path ${_test_exe_dir} ${_dict_lib})
   SET(_dict_lib_path ./${_dict_lib_path})

   # add the test with the relative dictionary path as a parameter
   REFLEX_ADD_TEST(${_name} ${_test_target} ${_dict_lib_path})

ENDMACRO (REFLEX_ADD_SINGLE_TEST _name)
