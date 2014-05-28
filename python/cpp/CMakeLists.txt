if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(python-cpp-cpp MACRO PyROOT_cpptests.py)
  ROOTTEST_ADD_TEST(python-cpp-advanced MACRO PyROOT_advancedtests.py)    
endif()
