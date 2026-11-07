if(pyroot)
  ROOTTEST_ADD_TEST(memory
                    MACRO PyROOT_memorytests.py
                    COPY_TO_BUILDDIR MemTester.C
                    PRECMD ${ROOT_root_CMD} -b -q -e .L\ MemTester.C+)
endif()
