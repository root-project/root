FIND_PACKAGE(CppUnit REQUIRED)
FIND_PACKAGE(PythonInterp REQUIRED)
FIND_PACKAGE(GCCXML REQUIRED)

#####################    config   ####################

SET(REFLEX_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

SET(_ctest_args_file ${CMAKE_BINARY_DIR}/Testing/Temporary/CTestArgs.log)
SET(_failed_tests_file ${CMAKE_BINARY_DIR}/Testing/Temporary/LastTestsFailed.log)

ADD_CUSTOM_TARGET(test_failed
                  COMMAND ${CMAKE_COMMAND} -D_failed_tests_file=${_failed_tests_file} -D_ctest_args_file=${_ctest_args_file} -P ${REFLEX_MODULE_DIR}/CreateFailedTestsArgs.cmake
                  COMMAND ${CMAKE_CTEST_COMMAND} -I ${_ctest_args_file}
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                  COMMENT "Running failed tests...")

#####################   sub dirs  ####################

MACRO_ADD_SUBDIRECTORIES(genreflex)
MACRO_ADD_SUBDIRECTORIES(implementation)
MACRO_ADD_SUBDIRECTORIES(iso-iec)
MACRO_ADD_SUBDIRECTORIES(regression)
