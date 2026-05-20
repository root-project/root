ROOTTEST_GENERATE_DICTIONARY(nextDict       next.h       LINKDEF nextLinkDef.h)

ROOTTEST_ADD_TEST(execNextTest
                  MACRO execNextTest.C
                  OUTREF NextTest.ref
                  DEPENDS nextDict-build
                  LABELS roottest regression cling)
