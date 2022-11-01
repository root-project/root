if(ROOT_pyroot_FOUND)
  if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(numba
                      MACRO PyROOT_numbatests.py
                      ENVIRONMENT LEGACY_PYROOT=${legacy_pyroot})
  endif()
endif()