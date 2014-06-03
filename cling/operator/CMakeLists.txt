ROOTTEST_ADD_TEST(roottest-cling-operator-runEqualTest
                  MACRO runEqualTest.C
                  OUTREF EqualTest.ref-cling
                  OUTCNV EqualTest_convert.sh
                  DEPENDS equal.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-operator-ParanTest
                  MACRO t01.C
                  OUTREF t01.ref
                  LABELS roottest regression cling)

# Verification of result skipped (see Makefile comment)
ROOTTEST_ADD_TEST(roottest-cling-operator-ConversionOp
                  MACRO runConversionOp.C
                  OUTCNV ConversionOp_convert.sh
                  DEPENDS ConversionOp.h
                  LABELS roottest regression cling)
