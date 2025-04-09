ROOTTEST_ADD_TEST(runEqualTest
                  MACRO runEqualTest.C
                  OUTREF_CINTSPECIFIC EqualTest.ref
                  OUTCNV EqualTest_convert.sh
                  DEPENDS equal.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(ParanTest
                  MACRO t01.C
                  OUTREF t01.ref
                  LABELS roottest regression cling)

if(ClingWorkAroundPrintfIssues AND ClingWorkAroundMissingUnloading)
  set(testcint TEST_OUTREF_CINTSPECIFIC ConversionOp.ref)
else()
  set(testcint "")
endif()

ROOTTEST_ADD_TEST(ConversionOp
                  MACRO runConversionOp.C
                  ${testcint}
                  OUTCNV ConversionOp_convert.sh
                  DEPENDS ConversionOp.h
                  LABELS roottest regression cling)
