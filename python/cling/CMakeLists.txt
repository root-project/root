if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(roottest-python-cling-api
                    MACRO runPyAPITest.C
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyAPITest.ref)

  ROOTTEST_ADD_TEST(roottest-python-cling-class MACRO
                    runPyClassTest.C
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyClassTest.ref)

  ROOTTEST_ADD_TEST(roottest-python-cling-cling
                    MACRO PyROOT_clingtests.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_clingtests.ref)
endif()
