if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(function
                    MACRO PyROOT_functiontests.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    PRECMD root -b -q -l -e .L\ InstallableFunction.C+
                                         -e .L\ GlobalFunction2.C+
                                         -e .L\ GlobalFunction3.C+
                    OUTREF PyROOT_functiontests.ref)
endif()
