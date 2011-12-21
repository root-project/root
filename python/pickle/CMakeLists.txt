if(ROOT_python_FOUND)
  ROOT_ADD_ROOTTEST(python-pickle-write MACRO PyROOT_writetests.py)
  ROOT_ADD_ROOTTEST(python-pickle-read MACRO PyROOT_readtests.py DEPENDS python-pickle-write-run)    
endif()
