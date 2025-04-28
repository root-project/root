if(ROOT_pyroot_FOUND)
  ROOTTEST_ADD_TEST(write
                    MACRO PyROOT_writetests.py
                    COPY_TO_BUILDDIR PickleTypes.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ PickleTypes.C+)

  ROOTTEST_ADD_TEST(read
                    MACRO PyROOT_readtests.py
                    DEPENDS write
                    ENVIRONMENT ROOFIT=${roofit})
endif()
