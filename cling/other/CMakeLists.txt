ROOTTEST_ADD_TEST(runfileClose
                  MACRO runfileClose.C
                  OUTREF fileClose.ref
                  DEPENDS fileClose.C
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(testCompile.C)

ROOTTEST_ADD_TEST(runsavannah54662
                  MACRO runsavannah54662.C
                  OUTREF savannah54662.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runsprintfselect
                  MACRO runsprintfselect.C
                  OUTREF sprintfselect.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(staticConstTest
                  MACRO staticConst.C
                  OUTREF staticConstTest.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runStaticDataMember
                  MACRO runStaticDataMember.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runmissingXFunc
                  MACRO runmissingXFunc.C
                  OUTREF missingXFunc.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(assertBaseMembers
                  MACRO assertBaseMembers.C
                  LABELS roottest regression cling)
