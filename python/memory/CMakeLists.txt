if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(memory
                    MACRO PyROOT_memorytests.py
                    COPY_TO_BUILDDIR MemTester.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ MemTester.C+
                    ENVIRONMENT EXP_PYROOT=${exp_pyroot}
                    ${PYTESTS_WILLFAIL})
endif()
