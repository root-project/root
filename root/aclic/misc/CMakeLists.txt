ROOTTEST_ADD_TEST(runrelpath
                  MACRO runrelpath.C
                  OUTREF relpath.ref
                  ${WILLFAIL_ON_WIN32}
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(assertmyfun
                  MACRO assertmyfun.cxx+
                  OUTREF assertmyfun.ref
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(assertROOT7027
                  MACRO assertROOT7027.C
                  OUTREF assertROOT7027.ref)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    ROOTTEST_ADD_TEST(addIncludePath
                      MACRO addIncludePath.C
                      COPY_TO_BUILDDIR addIncludePathTest.C addIncludePathGCCMajor.C)
endif()
