add_definitions(
  -DClingWorkAroundNoDotInclude
)

ROOTTEST_ADD_TEST(roottest-root-aclic-misc-runrelpath
                  MACRO runrelpath.C
                  OUTREF relpath.ref
                  DEPENDS script.C)

ROOTTEST_ADD_TEST(roottest-root-aclic-misc-assertmyfun
                  MACRO assertmyfun.cxx+
                  OUTREF assertmyfun.ref
                  DEPENDS script.C)
