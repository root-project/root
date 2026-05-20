if(pyroot)
  if(NOT MSVC OR "${CMAKE_GENERATOR_PLATFORM}" MATCHES "Win32" OR win_broken_tests)
    ROOTTEST_ADD_TEST(pythonizations
                      MACRO PyROOT_pythonizationtest.py
                      COPY_TO_BUILDDIR Pythonizables.C Pythonizables.h
                      PRECMD ${ROOT_root_CMD} -b -q -e .L\ Pythonizables.C+)
  endif()
endif()
