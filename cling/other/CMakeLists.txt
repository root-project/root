ROOTTEST_ADD_TEST(roottest-cling-other-runfileClose
                  MACRO runfileClose.C
                  OUTREF fileClose.ref
                  DEPENDS fileClose.C
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(testCompile.C)

ROOTTEST_ADD_TEST(roottest-cling-other-runsprintfselect
                  MACRO runsprintfselect.C
                  OUTREF sprintfselect.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-other-staticConstTest
                  MACRO staticConst.C
                  OUTREF staticConstTest.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-other-runStaticDataMember
                  MACRO runStaticDataMember.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-other-runsavannah54662
                  MACRO runsavannah54662.C
                  OUTREF savannah54662.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-other-runmissingXFunc
                  MACRO runmissingXFunc.C
                  OUTREF missingXFunc.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-other-assertBaseMembers
                  MACRO assertBaseMembers.C
                  LABELS roottest regression cling)
