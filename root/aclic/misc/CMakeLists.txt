ROOTTEST_ADD_TEST(runrelpath
                  MACRO runrelpath.C
                  OUTREF relpath.ref
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(assertmyfun
                  MACRO assertmyfun.cxx+
                  OUTREF assertmyfun.ref
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(assertROOT7027
                  MACRO assertROOT7027.C
                  OUTREF assertROOT7027.ref)
