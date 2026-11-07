if(pyroot)
  ROOTTEST_ADD_TEST(numba MACRO PyROOT_numbatests.py PYTHON_DEPS numba)
  file(COPY vec_lv.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()
