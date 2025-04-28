if(ROOT_pyroot_FOUND)
  ROOTTEST_ADD_TEST(memory
                    MACRO PyROOT_memorytests.py
                    COPY_TO_BUILDDIR MemTester.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ MemTester.C+)
endif()
