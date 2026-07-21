if(pyroot)
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(cpp
                    MACRO PyROOT_cpptests.py
                    COPY_TO_BUILDDIR Namespace.C
                    PRECMD ${ROOT_root_CMD} -b -q -e .L\ Namespace.C+)

  ROOTTEST_ADD_TEST(advanced
                    MACRO PyROOT_advancedtests.py
                    COPY_TO_BUILDDIR AdvancedCpp.C
                    PRECMD ${ROOT_root_CMD} -b -q -e .L\ AdvancedCpp.C+)
endif()
endif()
