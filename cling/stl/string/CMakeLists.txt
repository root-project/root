ROOTTEST_ADD_TEST(runcint
                  MACRO runcint.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(stringPassingText
                  MACRO run.C
                  OUTREF passing.ref
                  DEPENDS t01.C
                  LABELS roottest regression cling)

