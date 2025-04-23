if(ROOT_pyroot_FOUND)
  if(NOT MSVC OR "${CMAKE_GENERATOR_PLATFORM}" MATCHES "Win32" OR win_broken_tests)
    ROOTTEST_ADD_TEST(pythonizations
                      MACRO PyROOT_pythonizationtest.py
                      COPY_TO_BUILDDIR Pythonizables.C Pythonizables.h
                      PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ Pythonizables.C+
                      ENVIRONMENT LEGACY_PYROOT=${legacy_pyroot})
  endif()

  ROOTTEST_ADD_TEST(smartptr
                    MACRO PyROOT_smartptrtest.py
                    COPY_TO_BUILDDIR SmartPtr.C SmartPtr.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ SmartPtr.C+
                    ENVIRONMENT LEGACY_PYROOT=${legacy_pyroot})
endif()
