if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(cpp
                    MACRO PyROOT_cpptests.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    PRECMD root -b -q -l -e .L\ Namespace.C+ 
                                         -e .L\ PointerPassing.C+ 
                                         -e .L\ Namespace2.C+)

  ROOTTEST_ADD_TEST(advanced
                    MACRO PyROOT_advancedtests.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    PRECMD root -b -q -l -e .L\ AdvancedCpp.C+
                                         -e .L\ Template.C+)
endif()
