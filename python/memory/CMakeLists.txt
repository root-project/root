if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(memory
                    MACRO PyROOT_memorytests.py
                    PRECMD root -b -q -l -e .L\ MemTester.C+
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_memorytests.ref)
endif()
