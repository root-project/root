add_definitions(
  -DClingWorkAroundNoDotInclude
)

ROOTTEST_COMPILE_MACRO(script.C)

ROOTTEST_ADD_TEST(roottest-root-aclic-misc-runrelpath
                  MACRO runrelpath.C
                  OUTREF relpath.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                 )

ROOTTEST_ADD_TEST(roottest-root-aclic-misc-assertmyfun
                  MACRO assertmyfun.cxx+
                  OUTREF assertmyfun.ref
                  DEPENDS ${COMPILE_MACRO_TEST})
