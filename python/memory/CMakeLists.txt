if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(python-memory MACRO PyROOT_memorytests.py
                    PRECMD root -b -q -l -e .L\ MemTester.C+)
endif()
