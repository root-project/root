FIND_PACKAGE(CppUnit REQUIRED)
FIND_PACKAGE(PythonInterp REQUIRED)
FIND_PACKAGE(GCCXML REQUIRED)

#####################    config   ####################

SET(REFLEX_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

SET(_ctest_args_file ${CMAKE_BINARY_DIR}/Testing/Temporary/CTest.args)
SET(_failed_tests_file ${CMAKE_BINARY_DIR}/Testing/Temporary/LastTestsFailed.log)

CONFIGURE_FILE(${REFLEX_TEMPLATE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake @ONLY)

#####################    targets  ####################

# create a target "test_failed" to rerun all failed tests
ADD_CUSTOM_TARGET(test_failed
                  COMMAND ${CMAKE_COMMAND} -D_failed_tests_file=${_failed_tests_file} -D_ctest_args_file=${_ctest_args_file} -P ${REFLEX_MODULE_DIR}/CreateFailedTestsArgs.cmake
                  COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --switch=$(COLOR) --cyan "Running failed tests..."
                  COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process -I ${_ctest_args_file}
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# create a target "build_tests" to build all tests
IF (NOT REFLEX_BUILD_TESTS)
   ADD_CUSTOM_TARGET(build_tests)
ENDIF (NOT REFLEX_BUILD_TESTS)

#####################   sub dirs  ####################

MACRO_ADD_SUBDIRECTORIES(framework)
MACRO_ADD_SUBDIRECTORIES(genreflex)
MACRO_ADD_SUBDIRECTORIES(implementation)
MACRO_ADD_SUBDIRECTORIES(iso-iec)
MACRO_ADD_SUBDIRECTORIES(regression)
