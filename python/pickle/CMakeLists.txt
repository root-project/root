if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(roottest-python-pickle-write
                    MACRO PyROOT_writetests.py
                    PRECMD root -b -q -l -e .L\ PickleTypes.C+
                    OUTREF PyROOT_writetests.ref)

  ROOTTEST_ADD_TEST(roottest-python-pickle-read
                    MACRO PyROOT_readtests.py
                    OUTREF PyROOT_readtests.ref
                    DEPENDS python-pickle-write)    
endif()
