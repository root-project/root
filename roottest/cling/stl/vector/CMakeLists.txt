ROOTTEST_COMPILE_MACRO(t01.C)

ROOTTEST_ADD_TEST(Run
                  MACRO Run.C
                  DEPENDS ${COMPILE_MACRO_TEST}
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(t02
                  MACRO t02.C
                  WILLFAIL
                  LABELS roottest regression cling)
