if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(roottest-python-memory
                    MACRO PyROOT_memorytests.py
                    PRECMD root -b -q -l -e .L\ MemTester.C+
                    OUTREF PyROOT_memorytests.ref)
endif()
