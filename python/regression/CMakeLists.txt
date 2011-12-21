if(ROOT_python_FOUND)
  ROOT_ADD_ROOTTEST(python-regression MACRO PyROOT_regressiontests.py -b)
endif()
