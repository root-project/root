if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(pythonizations
                    MACRO PyROOT_pythonizationtest.py
                    COPY_TO_BUILDDIR Pythonizables.C Pythonizables.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ Pythonizables.C+
                    ${PYTESTS_WILLFAIL})

  ROOTTEST_ADD_TEST(smartptr
                    MACRO PyROOT_smartptrtest.py
                    COPY_TO_BUILDDIR SmartPtr.C SmartPtr.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ SmartPtr.C+
                    ${PYTESTS_WILLFAIL})
endif()
