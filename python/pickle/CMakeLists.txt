if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(python-pickle-write MACRO PyROOT_writetests.py
                    PRECMD root -b -q -l -e .L\ PickleTypes.C+)
  ROOTTEST_ADD_TEST(python-pickle-read MACRO PyROOT_readtests.py DEPENDS python-pickle-write-run)    
endif()
