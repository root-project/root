if(pyroot)
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(cpp
                    MACRO PyROOT_cpptests.py
                    COPY_TO_BUILDDIR Namespace.C PointerPassing.C  Namespace2.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ Namespace.C+
                                         -e .L\ PointerPassing.C+
                                         -e .L\ Namespace2.C+)

  ROOTTEST_ADD_TEST(advanced
                    MACRO PyROOT_advancedtests.py
                    COPY_TO_BUILDDIR AdvancedCpp.C Template.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ AdvancedCpp.C+
                                                     -e .L\ Template.C+)
endif()
endif()
