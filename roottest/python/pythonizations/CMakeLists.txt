if(pyroot)
  if(NOT MSVC OR "${CMAKE_GENERATOR_PLATFORM}" MATCHES "Win32" OR win_broken_tests)
    ROOTTEST_ADD_TEST(pythonizations
                      MACRO PyROOT_pythonizationtest.py)
  endif()
endif()
