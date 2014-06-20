if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(roottest-python-basic-basic
                    MACRO PyROOT_basictests.py
                    PRECMD root -b -q -l -e .L\ ArgumentPassingCompiled.C+
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_basictests.ref)

  ROOTTEST_ADD_TEST(roottest-python-basic-datatype
                    MACRO PyROOT_datatypetest.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_datatypetest.ref)

  ROOTTEST_ADD_TEST(roottest-python-basic-operator
                    MACRO PyROOT_operatortests.py
                    PRECMD root -b -q -l -e .L\ Operators.C+
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_operatortests.ref)

  ROOTTEST_ADD_TEST(roottest-python-basic-overload
                    MACRO PyROOT_overloadtests.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_overloadtests.ref)
endif()
