ROOTTEST_ADD_TEST(runlessThan
                  MACRO runlessThan.C
                  OUTREF lessThan.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runoperatorLessThanTemplate
                  MACRO runoperatorLessThanTemplate.C
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(scopeDict  scopeProblem.C LINKDEF linkdef.h)
ROOTTEST_GENERATE_DICTIONARY(scopeDict2 multiScope.C LINKDEF linkdef.h)

ROOTTEST_ADD_TEST(runReadbool
                  MACRO runReadbool.C
                  OUTREF Readbool.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(ptrconst
                  MACRO ptrconst.C+
                  OUTREF ptrconst.ref
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(TTestClass_h TTestClass.h)

ROOTTEST_ADD_TEST(runTTestClass
                  MACRO runTTestClass.C
                  OUTREF TTestClass.ref
                  DEPENDS ${GENERATE_DICTIONARY_TEST}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(semicolon
                  MACRO semicolon.C
                  MACROARG \"abc;xyz\"
                  OUTREF semicolon.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(assertManipulators
                  MACRO assertManipulators.C
                  OUTREF assertManipulators.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(assertSpaces
                  MACRO assertSpaces.C
                  OUTREF assertSpaces.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(execROOT6625
                  MACRO execROOT6625.C
                  OUTREF execROOT6625.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(assertUnnamedMacro
                  MACRO assertUnnamedMacro.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(assertROOT8773
                  MACRO assertROOT8773.C
                  LABELS roottest regression cling)
