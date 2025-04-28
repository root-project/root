if(ROOT_pyroot_FOUND)
  ROOTTEST_ADD_TEST(function
                    MACRO PyROOT_functiontests.py
                    COPY_TO_BUILDDIR InstallableFunction.C GlobalFunction.C GlobalFunction2.C GlobalFunction3.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ InstallableFunction.C+
                                             -e .L\ GlobalFunction2.C+
                                             -e .L\ GlobalFunction3.C+)
endif()
