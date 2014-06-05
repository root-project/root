if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(python-function MACRO PyROOT_functiontests.py
                    PRECMD root -b -q -l -e .L\ InstallableFunction.C+
                                         -e .L\ GlobalFunction2.C+
                                         -e .L\ GlobalFunction3.C+)
endif()
