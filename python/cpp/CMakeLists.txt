if(ROOT_python_FOUND)
  ROOT_ADD_ROOTTEST(python-cpp-cpp MACRO PyROOT_cpptests.py)
  ROOT_ADD_ROOTTEST(python-cpp-advanced MACRO PyROOT_advancedtests.py)    
endif()
