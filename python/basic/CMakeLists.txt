if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(python-basic-basic MACRO PyROOT_basictests.py
                    PRECMD root -b -q -l -e .L\ ArgumentPassingCompiled.C+)

  ROOTTEST_ADD_TEST(python-basic-datatype MACRO PyROOT_datatypetest.py)

  ROOTTEST_ADD_TEST(python-basic-operator MACRO PyROOT_operatortests.py
                    PRECMD root -b -q -l -e .L\ Operators.C+)

  ROOTTEST_ADD_TEST(python-basic-overload MACRO PyROOT_overloadtests.py)
endif()
