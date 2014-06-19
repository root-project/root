if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(roottest-python-cpp-cpp
                    MACRO PyROOT_cpptests.py
                    PRECMD root -b -q -l -e .L\ Namespace.C+ 
                                         -e .L\ PointerPassing.C+ 
                                         -e .L\ Namespace2.C+)

  ROOTTEST_ADD_TEST(roottest-python-cpp-advanced
                    MACRO PyROOT_advancedtests.py
                    PRECMD root -b -q -l -e .L\ AdvancedCpp.C+
                                         -e .L\ Template.C+)
endif()
