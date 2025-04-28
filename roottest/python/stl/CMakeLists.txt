if(ROOT_pyroot_FOUND)
  if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(stl
                      MACRO PyROOT_stltests.py
                      COPY_TO_BUILDDIR StlTypes.C
                      PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ StlTypes.C+)
  endif()
endif()

