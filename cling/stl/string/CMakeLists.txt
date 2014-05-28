# Add define to fix test.
add_definitions(-DClingWorkAroundMissingDynamicScope)

ROOTTEST_ADD_TEST(roottest-cling-stl-string-runcint
                  MACRO runcint.C
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(t01.C)

ROOTTEST_ADD_TEST(roottest-cling-stl-string-stringPassingText
                  MACRO run.C
                  OUTREF passing.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

