if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(roottest-python-cling-api
                    MACRO runPyAPITest.C
                    OUTREF PyAPITest.ref)

  ROOTTEST_ADD_TEST(roottest-python-cling-class MACRO
                    runPyClassTest.C
                    OUTREF PyClassTest.ref)

  ROOTTEST_ADD_TEST(roottest-python-cling-cling
                    MACRO PyROOT_clingtests.py
                    OUTREF PyROOT_clingtests.ref)
endif()
