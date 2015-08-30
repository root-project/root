if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(roottest-python-pythonizations-pythonizations
                    MACRO PyROOT_pythonizationtest.py
                    PRECMD root -b -q -l -e .L\ Pythonizables.C+
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_pythonizationstest.ref)
  ROOTTEST_ADD_TEST(roottest-python-pythonizations-smartptr
                    MACRO PyROOT_smartptrtest.py
                    PRECMD root -b -q -l -e .L\ SmartPtr.C+
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_smartptrtest.ref)
endif()
