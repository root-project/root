ROOTTEST_ADD_TEST(roottest-cling-parsing-runlessThan
                  MACRO runlessThan.C
                  OUTREF lessThan.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-parsing-runoperatorLessThanTemplate
                  MACRO runoperatorLessThanTemplate.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-parsing-runtestTable
                  MACRO runtestTable.C
                  OUTREF testTable.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(scopeDict  scopeProblem.C LINKDEF linkdef.h)
ROOTTEST_GENERATE_DICTIONARY(scopeDict2 multiScope.C LINKDEF linkdef.h)

ROOTTEST_ADD_TEST(roottest-cling-parsing-runReadbool
                  MACRO runReadbool.C
                  OUTREF Readbool.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-parsing-ptrconst
                  MACRO ptrconst.C+
                  OUTREF ptrconst.ref
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(TTestClass_h LINKDEF TTestClass.h)

ROOTTEST_ADD_TEST(roottest-cling-parsing-runTTestClass
                  MACRO runTTestClass.C
                  OUTREF TTestClass.ref
                  DEPENDS ${GENERATE_DICTIONARY_TEST}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-parsing-semicolon
                  MACRO semicolon.C
                  MACROARG \"abc;xyz\"
                  OUTREF semicolon.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-parsing-assertManipulators
                  MACRO assertManipulators.C
                  OUTREF assertManipulators.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-parsing-assertSpaces
                  MACRO assertSpaces.C
                  OUTREF assertSpaces.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-parsing-execROOT6625
                  MACRO execROOT6625.C
                  OUTREF execROOT6625.ref
                  LABELS roottest regression cling)

