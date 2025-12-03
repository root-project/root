if(ROOT_pyroot_FOUND)
  if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(numba
                      MACRO PyROOT_numbatests.py)
    file(COPY vec_lv.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()
