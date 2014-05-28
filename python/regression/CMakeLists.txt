if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(python-regression MACRO PyROOT_regressiontests.py -b)
endif()
