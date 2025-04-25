execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/ROOT-7775
                                                           ${CMAKE_CURRENT_BINARY_DIR}/ROOT-7775)

ROOTTEST_ADD_TEST(ROOT-7775
                  MACRO  execROOT-7775.C
                  OUTREF execROOT-7775.ref)

ROOTTEST_ADD_TESTDIRS()
