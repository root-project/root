if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(python-stl MACRO PyROOT_stltests.py
                    PRECMD root -b -q -l -e .L\ StlTypes.C+)
endif()
